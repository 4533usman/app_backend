import os
from datetime import datetime, timezone
from supabase import create_client, Client

_supabase: Client = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        _supabase = create_client(url, key)
    return _supabase


def create_job() -> str:
    """Insert a new job row with status='processing', return the job id."""
    sb = get_supabase()
    result = sb.table("api_jobs").insert({
        "status": "processing",
    }).execute()
    return result.data[0]["id"]


def update_job(job_id: str, **fields):
    """Update any columns on an api_jobs row."""
    sb = get_supabase()
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    sb.table("api_jobs").update(fields).eq("id", job_id).execute()


def insert_batch(job_id: str, segment_index: int, florence_captions: str, audio_path: str = None):
    """Insert one processed segment row into video_batches."""
    sb = get_supabase()
    sb.table("video_batches").insert({
        "job_id": job_id,
        "segment_index": segment_index,
        "florence_captions": florence_captions,
        "audio_path": audio_path,   # Supabase Storage path for edge function to use
    }).execute()


def upload_audio_segment(job_id: str, seg_label: str, local_path: str) -> str:
    """
    Upload a local audio file to Supabase Storage bucket 'audio-segments'.
    Returns the storage path: {job_id}/{seg_label}.mp3
    """
    sb = get_supabase()
    storage_path = f"{job_id}/{seg_label}.mp3"
    with open(local_path, "rb") as f:
        sb.storage.from_("audio-segments").upload(
            storage_path,
            f,
            file_options={"content-type": "audio/mpeg"}
        )
    return storage_path


def get_batches(job_id: str) -> list:
    """Return all video_batches rows for a job, ordered by segment_index."""
    sb = get_supabase()
    result = (
        sb.table("video_batches")
        .select("*")
        .eq("job_id", job_id)
        .order("segment_index")
        .execute()
    )
    return result.data


def get_job(job_id: str) -> dict | None:
    """Return the api_jobs row for a job_id."""
    sb = get_supabase()
    result = sb.table("api_jobs").select("*").eq("id", job_id).execute()
    return result.data[0] if result.data else None
