"""Batch evaluation routers."""
from fastapi import APIRouter

router = APIRouter()

@router.post("/run")
def run_batch():
    """Executes N sessions for evaluation purposes."""
    return {"message": "batch run started"}

@router.get("/{batch_id}/report")
def get_report(batch_id: str):
    """Obtains the evaluation report."""
    return {"report": {}}
