from django.db import models

# Create your models here.
from django.db import models

class LawCategory(models.Model):
    """Phân loại luật: Chính trị, Đất đai, Kinh tế, v.v."""
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

class Law(models.Model):
    """Mô hình lưu trữ thông tin bộ luật."""
    title = models.CharField(max_length=255)
    category = models.ForeignKey(LawCategory, on_delete=models.CASCADE)
    crawl_date = models.DateTimeField(auto_now_add=True)  # Thời gian crawl dữ liệu
    effective_start = models.DateField(null=True, blank=True)  # Ngày bắt đầu hiệu lực
    effective_end = models.DateField(null=True, blank=True)  # Ngày hết hiệu lực (nếu có)
    content = models.TextField()  # Nội dung của luật

    def __str__(self):
        return f"{self.title} ({self.category})"
