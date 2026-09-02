#!/usr/bin/env python3
"""
WebSocket Lambda Handler for Car Design Space Explorer.

Receives WebSocket messages from API Gateway, manages connection/session
state, routes queries to the Orchestrator Agent on AgentCore Runtime via
A2A protocol, and sends responses back through the WebSocket connection.

Architecture:
  - Synchronous path (API Gateway → Lambda): accepts the WebSocket message,
    stores it, returns 200 immediately, then self-invokes asynchronously.
  - Asynchronous path (Lambda → Lambda): performs the slow orchestrator call
    and posts the result back to the WebSocket connection.

This avoids the 29-second API Gateway WebSocket integration timeout.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
CONNECTIONS_TABLE = os.environ.get("CONNECTIONS_TABLE", "CarDesignWSConnections")
ORCHESTRATOR_RUNTIME_ARN = os.environ.get("ORCHESTRATOR_RUNTIME_ARN", "")
OAUTH_SECRET_NAME = os.environ.get(  # nosec B105 -- resource name, not secret value
    "OAUTH_SECRET_NAME",
    "car-design/agent-oauth-credentials",
)
LAMBDA_FUNCTION_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
GEOMETRY_S3_BUCKET = os.environ.get("GEOMETRY_S3_BUCKET", "")

# Clients (initialized lazily)
_dynamodb = None
_oauth_token: str | None = None
_oauth_token_expiry: float | None = None
_lambda_client = None
_s3_client = None


def _get_s3_client():
    """Lazy-init S3 client for presigned URL generation."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def _sanitize_incoming_message(text: str) -> str:
    """Strip presigned HTTPS S3 URLs from incoming messages, replacing with s3:// URIs.

    Users or frontend code may accidentally include presigned URLs (with
    embedded AWS credentials) in queries. These bloat the LLM context by
    ~2KB per URL, risk logging temporary credentials, and can push processing
    time past the Lambda timeout. Convert them to compact s3:// URIs instead.
    """
    import re
    from urllib.parse import urlparse

    def _to_s3_uri(match):
        url = match.group(0)
        try:
            parsed = urlparse(url)
            # hostname: <bucket>.s3.amazonaws.com or <bucket>.s3.<region>.amazonaws.com
            bucket = parsed.hostname.split(".s3.")[0]
            key = parsed.path.lstrip("/")
            if bucket and key:
                return f"s3://{bucket}/{key}"
        except Exception:
            pass
        return url  # leave unchanged if parsing fails

    # Match presigned S3 HTTPS URLs (with or without query string)
    pattern = r'https://[a-z0-9.-]+\.s3(?:\.[a-z0-9-]+)?\.amazonaws\.com/[^\s\'"<>]+'
    return re.sub(pattern, _to_s3_uri, text)


def _presign_s3_uris(text: str) -> str:
    """Replace s3:// URIs inside [STL] and [IMAGE] tags with presigned URLs.

    The orchestrator returns compact s3:// URIs in tags to keep A2A payloads
    small. This function converts them to presigned HTTPS URLs that the
    frontend can use directly.
    """
    import re

    def _presign(match):
        tag_type = match.group(1)  # STL or IMAGE
        s3_uri = match.group(2).strip()
        logger.info(f"[PRESIGN] Converting {tag_type} tag: {s3_uri}")
        try:
            clean_uri = s3_uri.replace("s3://", "")
            parts = clean_uri.split("/", 1)
            bucket, key = parts[0], parts[1]
            # Strip any trailing whitespace/newlines from key
            key = key.strip()
            logger.info(f"[PRESIGN] Bucket={bucket}, Key={key}")
            presigned = _get_s3_client().generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=3600,
            )
            logger.info(f"[PRESIGN] Success: presigned URL length={len(presigned)}")
            return f"[{tag_type}]{presigned}[/{tag_type}]"
        except Exception as e:
            logger.warning(f"[PRESIGN] Failed to presign {s3_uri}: {e}")
            return match.group(0)  # Return original on failure

    # Match [STL]s3://...[/STL] and [IMAGE]s3://...[/IMAGE]
    # Use [^\[\n]+ to avoid matching across newlines
    text = re.sub(r'\[(STL|IMAGE)\](s3://[^\[\n]+)\[/\1\]', _presign, text)
    # Also handle unclosed tags (truncated responses)
    text = re.sub(r'\[(STL|IMAGE)\](s3://[^\s\[\n]+)(?!\[)', _presign, text)
    return text


def _get_dynamodb():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    return _dynamodb


def _get_lambda_client():
    global _lambda_client
    if _lambda_client is None:
        _lambda_client = boto3.client("lambda", region_name=AWS_REGION)
    return _lambda_client


def _get_oauth_token() -> str:
    """Get OAuth token via client_credentials flow for agent invocation.

    Tokens are cached and refreshed 5 minutes before expiry (Cognito tokens
    last 60 minutes).
    """
    global _oauth_token, _oauth_token_expiry
    # Return cached token if still valid (with 5-min buffer)
    if _oauth_token and _oauth_token_expiry and time.time() < _oauth_token_expiry:
        return _oauth_token
    if not OAUTH_SECRET_NAME:
        raise RuntimeError("OAUTH_SECRET_NAME is required; OAuth cannot be skipped")
    try:
        import base64
        import http.client
        import ssl
        from urllib.parse import urlencode, urlsplit

        logger.info(f"Fetching OAuth token from secret: {OAUTH_SECRET_NAME}")
        secrets = boto3.client("secretsmanager", region_name=AWS_REGION)
        resp = secrets.get_secret_value(SecretId=OAUTH_SECRET_NAME)
        config = json.loads(resp["SecretString"])

        # Never send the OAuth client secret to an arbitrary endpoint if the
        # Secrets Manager value is accidentally or maliciously modified.
        token_url = str(config.get("token_url", ""))
        parsed_url = urlsplit(token_url)
        hostname = (parsed_url.hostname or "").lower()
        cognito_suffix = f".auth.{AWS_REGION}.amazoncognito.com"
        domain_prefix = (
            hostname[:-len(cognito_suffix)]
            if hostname.endswith(cognito_suffix)
            else ""
        )
        valid_prefix = bool(domain_prefix) and all(
            char.islower() or char.isdigit() or char == "-"
            for char in domain_prefix
        )
        if (
            parsed_url.scheme != "https"
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.port not in (None, 443)
            or parsed_url.path != "/oauth2/token"
            or parsed_url.query
            or parsed_url.fragment
            or not valid_prefix
        ):
            raise ValueError(
                "OAuth token_url must be the regional Cognito HTTPS token endpoint"
            )

        auth_b64 = base64.b64encode(
            f"{config['client_id']}:{config['client_secret']}".encode()
        ).decode()
        request_body = urlencode({
            "grant_type": "client_credentials",
            "scope": config.get("scope", ""),
        }).encode()
        logger.info("Requesting OAuth token from validated Cognito endpoint: %s", hostname)

        # Use a direct TLS connection to the validated host. Unlike urlopen,
        # HTTPSConnection cannot follow a redirect to another scheme or host.
        # Explicitly require normal system-CA certificate verification and TLS
        # 1.2+, independent of historical Python HTTPSConnection defaults.
        tls_context = ssl.create_default_context()
        tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        connection = http.client.HTTPSConnection(  # nosemgrep: python.lang.security.audit.httpsconnection-detected.httpsconnection-detected
            hostname,
            port=443,
            timeout=10,
            context=tls_context,
        )
        try:
            connection.request(
                "POST",
                "/oauth2/token",
                body=request_body,
                headers={
                    "Authorization": f"Basic {auth_b64}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            response = connection.getresponse()
            response_body = response.read()
            if not 200 <= response.status < 300:
                raise RuntimeError(
                    f"Cognito token endpoint returned HTTP {response.status}"
                )
            token_data = json.loads(response_body)
        finally:
            connection.close()

        _oauth_token = token_data["access_token"]
        # Cache for 55 minutes (Cognito tokens expire in 60 min)
        expires_in = token_data.get("expires_in", 3600)
        _oauth_token_expiry = time.time() + expires_in - 300  # 5-min buffer
        logger.info(f"OAuth token obtained successfully (expires in {expires_in}s)")
        return _oauth_token
    except Exception as e:
        logger.error(f"OAuth token retrieval failed: {e}", exc_info=True)
        _oauth_token = None
        _oauth_token_expiry = None
        return ""


def _get_agentcore_client():
    """Get an unsigned AgentCore client authenticated only with OAuth."""
    from botocore import UNSIGNED
    from botocore.config import Config

    token = _get_oauth_token()
    if not token:
        raise RuntimeError(
            "OAuth token unavailable; refusing insecure SigV4 fallback"
        )

    logger.info("Creating AgentCore client with OAuth bearer token (UNSIGNED)")
    client = boto3.client(
        "bedrock-agentcore",
        region_name=AWS_REGION,
        config=Config(signature_version=UNSIGNED),
    )

    def add_auth_header(request, **kwargs):
        request.headers["Authorization"] = f"Bearer {token}"

    client.meta.events.register_first("before-sign", add_auth_header)
    return client


_A2A_ERROR_STATES = {"failed", "canceled", "cancelled", "rejected"}


def _extract_a2a_task_error(task_data: dict) -> str | None:
    """Return a failed A2A task's status message, or None for non-failures."""
    if not isinstance(task_data, dict):
        return None

    status = task_data.get("status", {})
    if not isinstance(status, dict):
        return None

    state = str(status.get("state", "")).lower()
    if state not in _A2A_ERROR_STATES:
        return None

    message = status.get("message", {})
    parts = message.get("parts", []) if isinstance(message, dict) else []
    text_parts = [
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict) and part.get("text")
    ]
    return "".join(text_parts).strip() or f"Agent task {state}"


def _send_to_orchestrator(message: str, session_id: str) -> dict:
    """Send a message to the Orchestrator Agent via invoke_agent_runtime.

    For A2A protocol agents, the payload is passed through directly to POST /
    on the agent container. We send a proper A2A JSON-RPC 2.0 message/send
    envelope so the Strands A2AServer can process it without validation errors.
    """

    if not ORCHESTRATOR_RUNTIME_ARN:
        raise ValueError("ORCHESTRATOR_RUNTIME_ARN not configured")

    client = _get_agentcore_client()

    # A2A protocol: payload is passed through directly to POST / on the agent.
    # Must be a valid JSON-RPC 2.0 message/send request.
    a2a_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }
    payload = json.dumps(a2a_payload).encode()

    # Use a unique session ID per request to prevent AgentCore from injecting
    # stale session history. The orchestrator is stateless per request (it
    # clears messages via _clear_messages hook), but AgentCore's session
    # management re-injects old conversation history when the same session ID
    # is reused, causing ValidationException cascades and 300s timeouts.
    request_session_id = f"{session_id}-{uuid.uuid4().hex[:8]}"
    logger.info(f"Invoking orchestrator: ARN={ORCHESTRATOR_RUNTIME_ARN}, session={request_session_id}")

    response = client.invoke_agent_runtime(
        agentRuntimeArn=ORCHESTRATOR_RUNTIME_ARN,
        runtimeSessionId=request_session_id,
        payload=payload,
        qualifier="DEFAULT",
    )

    # Process response — may be streaming (text/event-stream) or JSON
    accumulated = ""
    content_type = response.get("contentType", "")
    logger.info(f"Response contentType: {content_type}")

    if "text/event-stream" in content_type:
        # Streaming response — accumulate text chunks
        # Also track A2A artifact text parts (concatenated from all parts)
        a2a_artifact_parts = []
        a2a_task_error = None
        for line in response["response"].iter_lines():
            if line:
                try:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        event_data = json.loads(data_str)
                    else:
                        event_data = json.loads(line_str)
                    if not isinstance(event_data, dict):
                        continue

                    # A2A streaming: inspect task state before extracting artifacts.
                    result = event_data.get("result", {})
                    if isinstance(result, dict):
                        task_data = result.get("task", result) if "task" in result else result
                        task_error = _extract_a2a_task_error(task_data)
                        if task_error:
                            a2a_task_error = task_error
                        artifacts = task_data.get("artifacts", [])
                        for artifact in artifacts:
                            for part in artifact.get("parts", []):
                                text = part.get("text", "")
                                if text:
                                    a2a_artifact_parts.append(text)

                    # Extract text from various streaming formats
                    chunk_text = ""
                    if "event" in event_data and "contentBlockDelta" in event_data["event"]:
                        delta = event_data["event"]["contentBlockDelta"]["delta"]
                        chunk_text = delta.get("text", "")
                    elif "text" in event_data:
                        chunk_text = event_data["text"]
                    elif isinstance(event_data, dict) and "data" in event_data:
                        chunk_text = str(event_data["data"])
                    if chunk_text:
                        accumulated += chunk_text
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Error processing stream line: {e}")

        if a2a_task_error:
            logger.error(f"Agent returned failed A2A task: {a2a_task_error}")
            return {"status": "error", "response": a2a_task_error}

        # A2A artifact text is the canonical complete answer assembled by the A2A
        # protocol from the agent's final response. Stream chunks are incremental
        # tokens and can include tool call echoes that inflate their length.
        # Always prefer artifact text when non-empty; fall back to chunks only
        # when artifacts are missing (e.g. non-A2A or malformed response).
        a2a_artifact_text = "".join(a2a_artifact_parts)
        logger.info(f"Accumulated chunks: {len(accumulated)} chars, first 300: {repr(accumulated[:300])}")
        logger.info(f"A2A artifact parts: {len(a2a_artifact_parts)} parts, joined {len(a2a_artifact_text)} chars, first 300: {repr(a2a_artifact_text[:300])}")
        if a2a_artifact_text:
            logger.info("Using A2A artifact text (canonical)")
            accumulated = a2a_artifact_text
        else:
            logger.info("Using accumulated stream chunks (no artifact parts found)")
    else:
        # Non-streaming — read all chunks
        content_parts = []
        for chunk in response.get("response", []):
            content_parts.append(chunk.decode("utf-8", errors="replace"))
        accumulated = "".join(content_parts)

        # Try to parse as JSON and extract text
        try:
            logger.info(f"Raw response (first 500 chars): {accumulated[:500]}")
            result = json.loads(accumulated)

            # A2A response: result.task.artifacts[].parts[].text
            if "result" in result:
                task_data = result["result"]
                if "task" in task_data:
                    task_data = task_data["task"]

                task_error = _extract_a2a_task_error(task_data)
                if task_error:
                    logger.error(f"Agent returned failed A2A task: {task_error}")
                    return {"status": "error", "response": task_error}

                artifacts = task_data.get("artifacts", [])
                text_parts = []
                for artifact in artifacts:
                    for part in artifact.get("parts", []):
                        text = part.get("text", "")
                        if not text and part.get("kind") == "text":
                            text = part.get("text", "")
                        if text:
                            text_parts.append(text)
                if text_parts:
                    accumulated = "".join(text_parts)

            # Simple response: {"result": "text"} or {"output": {...}}
            elif "output" in result:
                output = result["output"]
                if isinstance(output, dict):
                    accumulated = output.get("message", output.get("result", json.dumps(output)))
                else:
                    accumulated = str(output)

            # Error response
            elif "error" in result:
                error_msg = result["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", json.dumps(error_msg))
                logger.error(f"Agent returned error: {error_msg}")
                return {"status": "error", "response": str(error_msg)}

        except json.JSONDecodeError:
            # Not JSON — use raw text as-is
            pass

    resp_text = accumulated if accumulated else "No response from agent"
    logger.info(f"Final response length: {len(resp_text)} chars, first 200: {resp_text[:200]}")

    # Safety check: if the response looks like a raw JSON-RPC error, extract the message
    if resp_text.startswith("{") and '"error"' in resp_text[:200]:
        try:
            err_data = json.loads(resp_text)
            if "error" in err_data:
                err_msg = err_data["error"]
                if isinstance(err_msg, dict):
                    err_msg = err_msg.get("message", json.dumps(err_msg))
                logger.error(f"Agent returned JSON-RPC error: {err_msg}")
                return {"status": "error", "response": str(err_msg)}
        except (json.JSONDecodeError, Exception):
            pass

    return {
        "status": "success",
        "response": resp_text,
    }


def _post_to_connection(endpoint_url: str, connection_id: str, data: dict) -> None:
    """Send a message back to a WebSocket client."""
    client = boto3.client(
        "apigatewaymanagementapi",
        endpoint_url=endpoint_url,
        region_name=AWS_REGION,
    )
    try:
        client.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(data).encode(),
        )
    except client.exceptions.GoneException:
        logger.info(f"Connection {connection_id} is gone, cleaning up")
        _remove_connection(connection_id)


def _save_connection(connection_id: str, session_id: str) -> None:
    """Save WebSocket connection to DynamoDB."""
    table = _get_dynamodb().Table(CONNECTIONS_TABLE)
    table.put_item(Item={
        "connectionId": connection_id,
        "sessionId": session_id,
    })


def _remove_connection(connection_id: str) -> None:
    """Remove WebSocket connection from DynamoDB."""
    try:
        table = _get_dynamodb().Table(CONNECTIONS_TABLE)
        table.delete_item(Key={"connectionId": connection_id})
    except Exception as e:
        logger.warning(f"Failed to remove connection {connection_id}: {e}")


def _get_session_id(connection_id: str) -> str:
    """Get session ID for a connection, or create one."""
    try:
        table = _get_dynamodb().Table(CONNECTIONS_TABLE)
        resp = table.get_item(Key={"connectionId": connection_id})
        item = resp.get("Item")
        if item:
            return item.get("sessionId", str(uuid.uuid4()))
    except Exception:
        pass
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict, context: dict) -> dict:
    """Main Lambda handler for API Gateway WebSocket events.

    For sendMessage / $default routes, the handler uses a two-phase pattern:
      Phase 1 (synchronous, called by API Gateway):
        - Sends a "processing" indicator to the client
        - Asynchronously invokes *itself* with the orchestrator payload
        - Returns 200 immediately (well within the 29s APIGW timeout)
      Phase 2 (asynchronous, self-invoked):
        - Calls the orchestrator (may take 60-300s)
        - Posts the result back to the WebSocket connection
    """
    # Phase 2: async self-invocation for orchestrator call
    if event.get("_async_orchestrator"):
        return _handle_async_orchestrator(event)

    route_key = event.get("requestContext", {}).get("routeKey", "")
    connection_id = event.get("requestContext", {}).get("connectionId", "")
    domain = event.get("requestContext", {}).get("domainName", "")
    stage = event.get("requestContext", {}).get("stage", "")
    endpoint_url = f"https://{domain}/{stage}" if domain and stage else ""

    logger.info(f"Route: {route_key}, Connection: {connection_id}")

    if route_key == "$connect":
        return _handle_connect(event, connection_id)
    elif route_key == "$disconnect":
        return _handle_disconnect(connection_id)
    elif route_key in ("sendMessage", "$default"):
        return _handle_message(event, connection_id, endpoint_url)
    else:
        return {"statusCode": 400, "body": f"Unknown route: {route_key}"}


def _handle_connect(event: dict, connection_id: str) -> dict:
    """Handle WebSocket $connect — save connection, create session."""
    query = event.get("queryStringParameters") or {}
    session_id = query.get("sessionId", str(uuid.uuid4()))
    _save_connection(connection_id, session_id)
    logger.info(f"Connected: {connection_id}, session: {session_id}")
    return {"statusCode": 200}


def _handle_disconnect(connection_id: str) -> dict:
    """Handle WebSocket $disconnect — clean up connection."""
    _remove_connection(connection_id)
    logger.info(f"Disconnected: {connection_id}")
    return {"statusCode": 200}


def _upload_stl_to_s3(stl_data: dict) -> str:
    """Decode base64 STL and upload to S3. Returns the S3 URI."""
    import base64
    import time as _time

    file_name = stl_data.get("name", "uploaded.stl")
    b64 = stl_data.get("data", "")
    raw_bytes = base64.b64decode(b64)

    timestamp = int(_time.time())
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file_name)
    s3_key = f"geometries/uploaded_{timestamp}_{safe_name}"

    s3_client = boto3.client("s3", region_name=AWS_REGION)
    s3_client.put_object(
        Bucket=GEOMETRY_S3_BUCKET,
        Key=s3_key,
        Body=raw_bytes,
        ContentType="application/octet-stream",
    )
    s3_uri = f"s3://{GEOMETRY_S3_BUCKET}/{s3_key}"
    logger.info(f"Uploaded STL to {s3_uri} ({len(raw_bytes)} bytes)")
    return s3_uri


def _handle_message(event: dict, connection_id: str, endpoint_url: str) -> dict:
    """Handle incoming message — fire-and-forget to orchestrator via async self-invoke."""
    try:
        body = json.loads(event.get("body", "{}"))

        # --- Heartbeat ping: return 200 silently, no error to client ---
        if body.get("action") == "ping":
            return {"statusCode": 200}

        # --- STL file upload: upload to S3 and return path (no orchestrator call) ---
        stl_file = body.get("stlFile")
        if stl_file and stl_file.get("data"):
            try:
                _post_to_connection(endpoint_url, connection_id, {
                    "type": "processing",
                    "message": "Uploading geometry to storage...",
                })
                s3_uri = _upload_stl_to_s3(stl_file)
                _post_to_connection(endpoint_url, connection_id, {
                    "type": "stl_uploaded",
                    "s3Path": s3_uri,
                    "fileName": stl_file.get("name", "uploaded.stl"),
                    "fileSize": stl_file.get("size", 0),
                })
                logger.info(f"STL uploaded successfully: {s3_uri}")
                return {"statusCode": 200}
            except Exception as e:
                logger.error(f"STL upload failed: {e}", exc_info=True)
                _post_to_connection(endpoint_url, connection_id, {
                    "type": "error",
                    "error": f"STL upload failed: {str(e)}",
                })
                return {"statusCode": 500}

        message = body.get("message", body.get("payload", {}).get("message", ""))
        message = _sanitize_incoming_message(message)
        if not message:
            _post_to_connection(endpoint_url, connection_id, {
                "type": "error",
                "error": "No message provided",
            })
            return {"statusCode": 400}

        session_id = _get_session_id(connection_id)

        # Send processing indicator immediately
        _post_to_connection(endpoint_url, connection_id, {
            "type": "processing",
            "message": "Routing to design exploration agents...",
        })

        # Async self-invoke: fire the orchestrator call in a separate invocation
        # so we can return 200 to API Gateway within its 29s timeout.
        async_payload = {
            "_async_orchestrator": True,
            "message": message,
            "session_id": session_id,
            "connection_id": connection_id,
            "endpoint_url": endpoint_url,
        }
        _get_lambda_client().invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType="Event",  # async — returns immediately
            Payload=json.dumps(async_payload).encode(),
        )
        logger.info(f"Async orchestrator invocation dispatched for session {session_id}")

        return {"statusCode": 200}

    except Exception as e:
        logger.error(f"Message handling failed: {e}", exc_info=True)
        if endpoint_url and connection_id:
            _post_to_connection(endpoint_url, connection_id, {
                "type": "error",
                "error": f"Internal error: {str(e)}",
            })
        return {"statusCode": 500}


def _handle_async_orchestrator(event: dict) -> dict:
    """Phase 2: async handler that calls the orchestrator and posts results back."""
    message = event["message"]
    session_id = event["session_id"]
    connection_id = event["connection_id"]
    endpoint_url = event["endpoint_url"]

    logger.info(f"[ASYNC] Processing orchestrator call: session={session_id}, connection={connection_id}")

    try:
        result = _send_to_orchestrator(message, session_id)
        logger.info(f"[ASYNC] Orchestrator response status: {result.get('status')}")
        if result.get("status") == "error":
            logger.error(f"[ASYNC] Orchestrator error: {result.get('response', 'no details')}")
            # Send a user-friendly error instead of raw error text
            error_text = result.get("response", "The agent encountered an issue processing your request.")
            _post_to_connection(endpoint_url, connection_id, {
                "type": "error",
                "sessionId": session_id,
                "error": f"I encountered an issue while processing your request: {error_text}\n\nPlease try rephrasing your query or try again.",
            })
            return {"statusCode": 200}

        ws_payload = {
            "type": "response",
            "sessionId": session_id,
            **result,
        }

        # Convert s3:// URIs in [STL]/[IMAGE] tags to presigned URLs
        if "response" in ws_payload and isinstance(ws_payload["response"], str):
            ws_payload["response"] = _presign_s3_uris(ws_payload["response"])

        # API Gateway WebSocket has a 128KB frame limit.
        # Truncate if the payload is too large.
        payload_bytes = json.dumps(ws_payload).encode("utf-8")
        if len(payload_bytes) > 125_000:
            logger.warning(f"[ASYNC] Response too large ({len(payload_bytes)} bytes), truncating")
            resp_text = result.get("response", "")
            # Trim the response text to fit within limits
            max_text = 120_000
            if len(resp_text.encode("utf-8")) > max_text:
                resp_text = resp_text[:max_text] + "\n\n[Response truncated due to size limits]"
            ws_payload = {
                "type": "response",
                "sessionId": session_id,
                "status": result.get("status", "success"),
                "response": resp_text,
            }

        _post_to_connection(endpoint_url, connection_id, ws_payload)

    except Exception as e:
        logger.error(f"[ASYNC] Orchestrator call failed: {e}", exc_info=True)
        try:
            _post_to_connection(endpoint_url, connection_id, {
                "type": "error",
                "error": f"Agent processing failed: {str(e)}",
            })
        except Exception as post_err:
            logger.error(f"[ASYNC] Failed to post error to connection: {post_err}")

    return {"statusCode": 200}
