from django.shortcuts import render
from .models import Keyword

def index(request):

    if request.method == "POST":

        keyword = request.POST.get("keyword")

        if keyword and keyword.strip():

            Keyword.objects.create(
                keyword=keyword
            )

    data = Keyword.objects.all().order_by('-created_at')

    return render(request, "KW/html/index.html", {
        "data": data
    })