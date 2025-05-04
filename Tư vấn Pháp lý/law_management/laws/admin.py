from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Law, LawCategory

@admin.register(LawCategory)
class LawCategoryAdmin(admin.ModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)

@admin.register(Law)
class LawAdmin(admin.ModelAdmin):
    list_display = ("title", "category", "crawl_date", "effective_start", "effective_end")
    list_filter = ("category", "effective_start", "effective_end")
    search_fields = ("title", "content")
    date_hierarchy = "crawl_date"