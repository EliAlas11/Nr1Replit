from fastapi import APIRouter, HTTPException
from app.schemas import FeedbackIn, FeedbackOut, FeedbackList, Message
from ..controllers.feedback_controller import submit_feedback, get_feedback

router = APIRouter(prefix="/api/v1/feedback", tags=["Feedback"])

@router.post("/", response_model=Message)
def submit(feedback: FeedbackIn):
    result = submit_feedback(feedback)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return Message(message=result["message"])

@router.get("/", response_model=FeedbackList)
def get():
    feedback_list = get_feedback()
    return FeedbackList(feedback=feedback_list)
