from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .rag_service import generate_answer


def home(request):
    return JsonResponse({
        "message": "AI Customer Retention Assistant API is running"
    })


@csrf_exempt
def ask(request):

    if request.method != "POST":
        return JsonResponse({
            "error": "Only POST method is allowed"
        }, status=405)

    try:

        data = json.loads(request.body)

        question = data.get("question")

        if not question:
            return JsonResponse({
                "error": "Question is required"
            }, status=400)

        result = generate_answer(question)

        return JsonResponse({
            "question": question,
            "answer": result["answer"],
            "retrieved_documents": result["retrieved_documents"]
        })

    except Exception as e:

        return JsonResponse({
            "error": str(e)
        }, status=500)