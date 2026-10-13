"""Secrets from AWS Systems Manager Parameter Store, read once when the container starts. Standard library only.

    SSM_PARAMETER_PATH=/voice2decisions/     # every parameter under this path becomes an environment variable:
                                             #   /voice2decisions/EXTRACT_MODEL_API_KEY -> EXTRACT_MODEL_API_KEY

SecureString values are decrypted (WithDecryption). A parameter only fills a variable that is missing or empty, so a
value set in .env still wins. Credentials come from the environment (AWS_ACCESS_KEY_ID, ...) or else from the EC2
instance role through the instance metadata service (IMDSv2); the region from AWS_REGION / AWS_DEFAULT_REGION, else
the instance's. If the path is set and the parameters can't be read, the server doesn't start, so a missing secret is
never silent. Nothing here is written to disk or logged except the names of the variables that were filled.
"""
import datetime
import hashlib
import hmac
import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger(__name__)
IMDS = "http://169.254.169.254"


def _imds(path: str, token: str) -> str:
    req = urllib.request.Request(f"{IMDS}{path}", headers={"X-aws-ec2-metadata-token": token})
    with urllib.request.urlopen(req, timeout=2) as resp:
        return resp.read().decode()


def _imds_token() -> str:
    req = urllib.request.Request(f"{IMDS}/latest/api/token", method="PUT",
                                 headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"})
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.read().decode()
    except OSError as exc:
        # From inside a container the token reply needs two network hops; EC2's default allows one.
        raise RuntimeError(f"no instance role credentials ({exc}). On EC2, allow containers to reach the metadata "
                           "service: aws ec2 modify-instance-metadata-options --instance-id <id> "
                           "--http-put-response-hop-limit 2 --http-tokens required") from None


def credentials(env) -> tuple[dict, str]:
    """({access_key, secret_key, token}, region)."""
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or ""
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        creds = {"access_key": env["AWS_ACCESS_KEY_ID"], "secret_key": env["AWS_SECRET_ACCESS_KEY"],
                 "token": env.get("AWS_SESSION_TOKEN", "")}
        token = None
    else:
        token = _imds_token()
        role = _imds("/latest/meta-data/iam/security-credentials/", token).split()[0]
        doc = json.loads(_imds(f"/latest/meta-data/iam/security-credentials/{role}", token))
        creds = {"access_key": doc["AccessKeyId"], "secret_key": doc["SecretAccessKey"], "token": doc.get("Token", "")}
    if not region:
        region = _imds("/latest/meta-data/placement/region", token or _imds_token())
    return creds, region


def sign(method: str, url_host: str, path: str, query: str, headers: dict, body: bytes, creds: dict, region: str,
         service: str, now: datetime.datetime) -> dict:
    """AWS Signature Version 4: returns the headers to send (the given ones plus x-amz-date, token, Authorization)."""
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    headers = {**headers, "host": url_host, "x-amz-date": amz_date}
    if creds.get("token"):
        headers["x-amz-security-token"] = creds["token"]
    canon = {k.lower(): " ".join(str(v).split()) for k, v in headers.items()}
    signed = ";".join(sorted(canon))
    request = "\n".join([method, path, query, *(f"{k}:{canon[k]}" for k in sorted(canon)), "", signed,
                         hashlib.sha256(body).hexdigest()])
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(request.encode()).hexdigest()])
    key = f"AWS4{creds['secret_key']}".encode()
    for part in (day, region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    headers["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={creds['access_key']}/{scope}, "
                                f"SignedHeaders={signed}, Signature={signature}")
    return headers


def parameters(path: str, env) -> dict[str, str]:
    """Every parameter under `path` (recursive, decrypted), keyed by its name below the path."""
    creds, region = credentials(env)
    endpoint = env.get("SSM_ENDPOINT") or f"https://ssm.{region}.amazonaws.com"
    host = endpoint.split("://", 1)[1].rstrip("/")
    prefix = path.rstrip("/") + "/"
    out, next_token = {}, None
    while True:
        payload = {"Path": prefix.rstrip("/") or "/", "Recursive": True, "WithDecryption": True}
        if next_token:
            payload["NextToken"] = next_token
        body = json.dumps(payload).encode()
        headers = sign("POST", host, "/", "", {"content-type": "application/x-amz-json-1.1",
                                               "x-amz-target": "AmazonSSM.GetParametersByPath"},
                       body, creds, region, "ssm", datetime.datetime.now(datetime.timezone.utc))
        req = urllib.request.Request(endpoint.rstrip("/") + "/", data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                page = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Parameter Store refused GetParametersByPath on {prefix}: "
                               f"{exc.code} {exc.read()[:300].decode(errors='replace')}") from None
        for p in page.get("Parameters", []):
            out[p["Name"][len(prefix):] if p["Name"].startswith(prefix) else p["Name"].rsplit("/", 1)[-1]] = p["Value"]
        next_token = page.get("NextToken")
        if not next_token:
            return out


def load_into(env=os.environ) -> list[str]:
    """Fill missing or empty environment variables from SSM_PARAMETER_PATH. Returns the names that were filled."""
    path = env.get("SSM_PARAMETER_PATH", "")
    if not path:
        return []
    filled = []
    for name, value in parameters(path, env).items():
        var = name.replace("/", "_").replace("-", "_").upper()
        if not env.get(var):
            env[var] = value
            filled.append(var)
    log.info("from Parameter Store %s: %s", path, ", ".join(filled) or "nothing new")
    return filled
