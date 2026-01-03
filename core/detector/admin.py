from django.contrib import admin
from .models import PredictionHistory

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "image_name", "result", "score", "created_at")
    list_filter = ("result", "created_at")
    search_fields = ("user__username", "image_name")
