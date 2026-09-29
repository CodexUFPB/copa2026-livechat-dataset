#!/usr/bin/env python3
"""
Pseudonymization pipeline for the CazéTV YouTube live chat dataset
(Brazil matches, FIFA World Cup 2026).

Usage:
    python anonymize.py --input RAW_DIR --output data/ --key PATH_TO_SECRET_KEY \
                        [--review review.csv] [--report-dir DIR]

The secret key file must NEVER be published. With the same key and the same
raw files, the output is fully reproducible.
"""
import argparse, glob, hashlib, hmac, os, re, secrets, sys
import pandas as pd

# Accounts of organizations (not natural persons) kept with their real handle.
INSTITUTIONAL = {"@cazetv", "@itau", "@mercadolivreoficial"}

# Automatic YouTube membership notices (not user speech): removed.
SYSTEM_PATTERNS = [
    r"^@\S+ just became a member!$",
    r"^@\S+ received a gift membership by @\S+$",
    r"^@\S+ gifted \d+ .*memberships?$",
    r"^@\S+ celebrates \d+ months? of membership$",
]
SYSTEM_RE = re.compile("|".join(SYSTEM_PATTERNS))

MENTION_RE = re.compile(r"@[\w.\-]+")
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w\-]+\.(?:ly|com|com\.br|net|org)/\S*)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b")
# Brazilian phone numbers written with a separator or +55 (avoids score spam like 6767676767)
PHONE_RE = re.compile(r"(\+?55\s?)?\(?\b\d{2}\)?[\s.]?9?\d{4}[\s.\-]\d{4}\b")

# Self-identification cues flagged for manual review
FLAG_RE = re.compile(
    r"(?:meu nome|me chamo|fala meu nome|manda (?:um )?salve|me segue|segue (?:a[ií]|l[aá])|"
    r"aqui [ée] (?:o|a) [A-ZÁÉÍÓÚ]|meu insta|meu zap|whats ?app)", re.I)
NAME_AFTER_RE = re.compile(
    r"((?i:meu nome (?:é|e|eh)|me chamo|aqui [ée] (?:o|a))\s+)"
    r"((?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+){0,3})")


def load_key(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
        print(f"[info] new secret key created at {path}. Keep it private and backed up.")
    return bytes.fromhex(open(path).read().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--review", default=None, help="CSV with manual review decisions")
    ap.add_argument("--report-dir", default=None)
    a = ap.parse_args()
    key = load_key(a.key)
    report_dir = a.report_dir or a.output
    os.makedirs(a.output, exist_ok=True); os.makedirs(report_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.input, "jogo*.csv")))
    if not files:
        sys.exit("no input files found")
    frames = []
    for i, f in enumerate(files, 1):
        d = pd.read_csv(f, encoding="utf-8-sig", dtype={"message": str})
        name = os.path.basename(f)[:-4]            # jogo01-13-06-brasil-marrocos
        d["game_id"] = f"J{i:02d}"
        d["game"] = "-".join(name.split("-")[3:])  # brasil-marrocos
        d["source_file"] = name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    stats = {"messages_raw": len(df), "authors_raw": df.channel_id.nunique()}

    # ---- pseudonyms, keyed on the stable channel_id ----
    def pseudo(channel_id):
        return "u_" + hmac.new(key, channel_id.encode(), hashlib.sha256).hexdigest()[:12]

    handle_by_channel = df.groupby("channel_id").author.last()
    pid = {}
    for ch, h in handle_by_channel.items():
        pid[ch] = h if h.lower() in INSTITUTIONAL else pseudo(ch)
    if len(set(pid.values())) != len(pid):
        sys.exit("pseudonym collision, increase hash length")
    handle_to_pid = {}
    for ch, h in df[["channel_id", "author"]].drop_duplicates().itertuples(index=False):
        handle_to_pid[h.lower()] = pid[ch]

    # ---- drop system membership notices ----
    msg = df.message.fillna("")
    is_system = msg.str.match(SYSTEM_RE)
    stats["system_notices_removed"] = int(is_system.sum())
    df = df[~is_system].copy()

    # ---- text redaction ----
    counters = {"mentions_known": 0, "mentions_unknown": 0, "urls": 0, "emails": 0, "phones": 0}

    def repl_mention(m):
        tok = m.group(0)
        stripped = tok.rstrip(".-")
        tail = tok[len(stripped):]
        low = stripped.lower()
        if low in INSTITUTIONAL:
            return tok
        if low in handle_to_pid:
            counters["mentions_known"] += 1
            return "@" + handle_to_pid[low] + tail
        counters["mentions_unknown"] += 1
        return "@usuario" + tail

    def clean(text, keep_urls):
        if not isinstance(text, str) or text == "":
            return text
        text, n = EMAIL_RE.subn("[EMAIL]", text); counters["emails"] += n
        if not keep_urls:
            text, n = URL_RE.subn("[URL]", text); counters["urls"] += n
        text, n = PHONE_RE.subn("[TELEFONE]", text); counters["phones"] += n
        text = MENTION_RE.sub(repl_mention, text)
        return text

    inst = df.author.str.lower().isin(INSTITUTIONAL)
    df["message"] = [clean(t, k) for t, k in zip(df.message, inst)]
    stats.update(counters)

    # ---- final columns and ids ----
    df["author_id"] = df.channel_id.map(pid)
    df = df.sort_values(["game_id", "timestamp"], kind="stable")
    df["message_id"] = df.groupby("game_id").cumcount().add(1).map("{:06d}".format)
    df["message_id"] = df.game_id + "_" + df.message_id

    # ---- manual review ----
    if a.review and os.path.exists(a.review):
        rv = pd.read_csv(a.review, encoding="utf-8-sig", dtype=str).fillna("")
        rv = rv[rv.acao.str.strip().str.lower() == "redigir"]
        repl = dict(zip(rv.message_id, rv.texto_revisado))
        df.loc[df.message_id.isin(repl), "message"] = df.message_id.map(repl)
        stats["manual_redactions"] = len(repl)
    else:
        flagged = df[df.message.fillna("").str.contains(FLAG_RE) & ~inst].copy()
        flagged["sugestao"] = flagged.message.str.replace(NAME_AFTER_RE, r"\1[NOME]", regex=True)
        out = flagged[["message_id", "message", "sugestao"]].rename(columns={"sugestao": "texto_revisado"})
        out.insert(1, "acao", "")
        path = os.path.join(report_dir, "revisao_manual.csv")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        stats["flagged_for_review"] = len(out)
        print(f"[info] {len(out)} messages flagged for review -> {path}")

    cols = ["message_id", "game_id", "game", "timestamp", "author_id", "message",
            "is_owner", "is_moderator", "is_verified", "is_member"]
    for (gid, src), g in df.groupby(["game_id", "source_file"]):
        g[cols].to_csv(os.path.join(a.output, f"{src}.csv"), index=False, encoding="utf-8")
    df[cols].to_csv(os.path.join(a.output, "all_games.csv"), index=False, encoding="utf-8")

    stats["messages_final"] = len(df)
    stats["authors_final"] = df.author_id.nunique()
    stats["authors_in_multiple_games"] = int((df.groupby("author_id").game_id.nunique() > 1).sum())
    pd.Series(stats).to_csv(os.path.join(report_dir, "anonymization_report.csv"), header=["value"])
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
