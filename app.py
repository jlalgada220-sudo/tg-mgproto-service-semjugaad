import os
import re
import tempfile
import mimetypes
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pyrogram import Client

try:
    import gradio as gr
except ImportError:
    gr = None

# ==============================================================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==============================================================================
API_ID_RAW = os.environ.get("API_ID", "0")
API_ID = int(API_ID_RAW) if API_ID_RAW.isdigit() else 0
API_HASH = os.environ.get("API_HASH", "").strip()
BOT_TOKENS_RAW = os.environ.get("BOT_TOKEN", "").strip()
CHANNEL_ID = int(os.environ.get("CHANNEL_ID", "0"))
SECRET_KEY = os.environ.get("SECRET_KEY", "").strip()

# Clean token list supporting 1 or multiple tokens separated by commas
BOT_TOKENS: List[str] = [token.strip() for token in BOT_TOKENS_RAW.split(",") if token.strip()]

# 1MB chunk size for memory-efficient streaming and uploading
CHUNK_SIZE = 1024 * 1024

# ==============================================================================
# IN-MEMORY CLIENT POOL (ROUND-ROBIN LOAD BALANCING)
# ==============================================================================
clients: List[Client] = []
client_index: int = 0


def get_client() -> Client:
    """Return next available Pyrogram client in round-robin sequence."""
    global client_index
    if not clients:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No Telegram MTProto bot clients currently connected. Check API_ID, API_HASH and BOT_TOKEN."
        )
    client = clients[client_index % len(clients)]
    client_index += 1
    return client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize in-memory MTProto clients on startup and shut down cleanly."""
    if not API_ID or not API_HASH or not BOT_TOKENS:
        print("[!] WARNING: API_ID, API_HASH, or BOT_TOKEN missing. Please configure environment variables.")

    for idx, token in enumerate(BOT_TOKENS):
        try:
            client = Client(
                name=f"mtproto_bot_{idx}",
                api_id=API_ID,
                api_hash=API_HASH,
                bot_token=token,
                in_memory=True,     # Keeps session exclusively in RAM - no disk session lock collisions
                no_updates=True     # Disables update dispatcher to maximize streaming performance
            )
            await client.start()
            clients.append(client)
            me = await client.get_me()
            print(f"[*] Connected MTProto Client #{idx + 1}: @{me.username} ({me.id})")
        except Exception as err:
            print(f"[!] Failed to connect client #{idx + 1} with token {token[:10]}...: {err}")

    yield

    # Clean shutdown of all clients
    print("[*] Stopping Telegram MTProto client pool...")
    for idx, client in enumerate(clients):
        try:
            if client.is_connected:
                await client.stop()
                print(f"[*] Disconnected client #{idx + 1}")
        except Exception as stop_err:
            print(f"[!] Error stopping client #{idx + 1}: {stop_err}")


# ==============================================================================
# FASTAPI APPLICATION SETUP
# ==============================================================================
app = FastAPI(
    title="Telegram MTProto Media Microservice",
    description="High-performance, 2GB streaming and storage service bypassing 20MB bot limits",
    version="1.0.0",
    lifespan=lifespan
)

# Enable Full CORS for browser video streaming and cross-origin PDF viewing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Disposition"]
)


# ==============================================================================
# ROUTES & ENDPOINTS
# ==============================================================================
@app.get("/api")
async def root():
    return {
        "service": "Telegram MTProto Media Streaming Service",
        "status": "healthy",
        "bots_connected": len(clients),
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint to verify bot pool connection status."""
    return {
        "status": "ok",
        "bots_connected": len(clients),
        "channel_configured": bool(CHANNEL_ID)
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_api_key: str = Header(None, alias="x-api-key")
):
    """
    Accepts multipart file uploads up to 2GB.
    Streams incoming file in 1MB chunks to disk (/tmp), uploads to Telegram channel,
    removes temp file, and returns the channel message_id.
    """
    if not SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: SECRET_KEY is not defined"
        )

    if x_api_key != SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid x-api-key header"
        )

    if not CHANNEL_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: CHANNEL_ID is not configured"
        )

    client = get_client()

    # Stream incoming multipart request to /tmp in 1MB chunks to keep memory footprint minimal
    temp_dir = tempfile.gettempdir()
    safe_filename = os.path.basename(file.filename or "uploaded_file.bin")
    temp_file_path = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}_{safe_filename}")

    try:
        total_bytes = 0
        with open(temp_file_path, "wb") as buffer:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                buffer.write(chunk)
                total_bytes += len(chunk)

        # Upload to target Telegram channel via MTProto send_document
        sent_msg = await client.send_document(
            chat_id=CHANNEL_ID,
            document=temp_file_path,
            file_name=safe_filename,
            caption=f"SemJugaad Media Archive: {safe_filename} ({round(total_bytes / (1024 * 1024), 2)} MB)"
        )

        media = sent_msg.document or sent_msg.video or sent_msg.audio or sent_msg.photo
        ret_file_name = getattr(media, "file_name", safe_filename)
        ret_mime_type = getattr(media, "mime_type", file.content_type or "application/octet-stream")
        ret_file_size = getattr(media, "file_size", total_bytes)

        return {
            "status": "success",
            "message_id": sent_msg.id,
            "file_name": ret_file_name,
            "mime_type": ret_mime_type,
            "file_size": ret_file_size
        }

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Telegram MTProto upload error: {str(exc)}"
        )
    finally:
        # Guarantee cleanup of temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


@app.get("/stream/{message_id}")
async def stream_media(
    message_id: int,
    request: Request,
    download: bool = False
):
    """
    Streams media from Telegram channel with full HTTP 206 Partial Content Range support.
    Enables seeking in HTML5 video players and responsive PDF page rendering.
    """
    if not CHANNEL_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: CHANNEL_ID is not configured"
        )

    client = get_client()

    try:
        msg = await client.get_messages(chat_id=CHANNEL_ID, message_ids=message_id)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch Telegram message {message_id}: {str(err)}"
        )

    if not msg or msg.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {message_id} not found in channel"
        )

    media = msg.document or msg.video or msg.audio or (msg.photo and msg.photo[-1])
    if not media:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No streamable media item found in this message"
        )

    file_size = getattr(media, "file_size", 0)
    file_name = getattr(media, "file_name", f"media_{message_id}.bin")
    mime_type = getattr(media, "mime_type", None)

    if not mime_type:
        mime_type, _ = mimetypes.guess_type(file_name)
    if not mime_type:
        mime_type = "application/octet-stream"

    # Clean Content-Disposition header
    disposition_type = "attachment" if download else "inline"
    safe_ascii_name = re.sub(r'[^\w\.-]', '_', file_name)
    disposition_header = f'{disposition_type}; filename="{safe_ascii_name}"'

    range_header = request.headers.get("range")

    # ==========================================================================
    # CASE 1: HTTP 206 Partial Content (Range Request - video scrub / PDF seek)
    # ==========================================================================
    if range_header:
        range_match = re.match(r"^bytes=(\d+)-(\d*)$", range_header.strip())
        if not range_match:
            raise HTTPException(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail="Malformed Range header format. Expected 'bytes=start-end'."
            )

        start = int(range_match.group(1))
        end_str = range_match.group(2)
        end = int(end_str) if end_str else file_size - 1

        if start >= file_size or end >= file_size or start > end:
            return Response(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                headers={
                    "Content-Range": f"bytes */{file_size}",
                    "Accept-Ranges": "bytes"
                }
            )

        content_length = end - start + 1

        async def range_chunk_generator():
            start_chunk = start // CHUNK_SIZE
            first_chunk_skip = start % CHUNK_SIZE
            bytes_remaining = content_length

            async for chunk in client.stream_media(msg, offset=start_chunk):
                if bytes_remaining <= 0:
                    break

                # Discard leading bytes from first chunk if offset is not chunk-aligned
                if first_chunk_skip > 0:
                    chunk = chunk[first_chunk_skip:]
                    first_chunk_skip = 0

                # Truncate final chunk if it exceeds requested Range end
                if len(chunk) > bytes_remaining:
                    chunk = chunk[:bytes_remaining]

                bytes_remaining -= len(chunk)
                yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": mime_type,
            "Content-Disposition": disposition_header,
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length, Content-Disposition",
        }

        return StreamingResponse(
            range_chunk_generator(),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=headers
        )

    # ==========================================================================
    # CASE 2: HTTP 200 OK (Full File Stream)
    # ==========================================================================
    async def full_chunk_generator():
        async for chunk in client.stream_media(msg):
            yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": mime_type,
        "Content-Disposition": disposition_header,
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length, Content-Disposition",
    }

    return StreamingResponse(
        full_chunk_generator(),
        status_code=status.HTTP_200_OK,
        headers=headers
    )


# ==============================================================================
# GRADIO DASHBOARD INTEGRATION (Optional fallback for Hugging Face or local view)
# ==============================================================================
if gr is not None:
    try:
        with gr.Blocks(title="SemJugaad Media Engine") as demo:
            gr.Markdown("""
            # ⚡ SemJugaad Telegram MTProto Media Microservice
            *High-performance 2GB media storage & HTTP 206 Range streaming engine.*

            ### 🟢 Service Status
            The MTProto streaming engine is **active and listening** for incoming requests from SemJugaad.

            ### 📡 API Endpoints
            * `GET /health` — Service health & connected bots count
            * `POST /upload` — Multipart 2GB file upload (requires `x-api-key` header)
            * `GET /stream/{message_id}` — Byte-range video seeking and PDF viewer streaming
            """)

        # Mount Gradio dashboard at root "/" while serving all FastAPI routes
        app = gr.mount_gradio_app(app, demo, path="/")
    except Exception as e:
        print(f"[!] Gradio UI mounting skipped: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
