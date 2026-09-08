from django.urls import path
from .views import home, ask

urlpatterns = [
    path("", home, name="home"),
    path("api/ask/", ask, name="ask"),
]