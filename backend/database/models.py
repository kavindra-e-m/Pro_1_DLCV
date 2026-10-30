from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database.db import Base

def now_utc(): return datetime.now(timezone.utc)

class Product(Base):
    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("source_dataset", "source_product_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_dataset: Mapped[str] = mapped_column(String(40), index=True)
    source_product_id: Mapped[str] = mapped_column(String(160), index=True)
    product_identity: Mapped[str] = mapped_column(String(200), index=True)
    category: Mapped[str | None] = mapped_column(String(160), nullable=True)
    brand: Mapped[str | None] = mapped_column(String(200), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    images: Mapped[list["Image"]] = relationship(back_populates="product", cascade="all, delete-orphan")

class Image(Base):
    __tablename__ = "images"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    image_path: Mapped[str] = mapped_column(Text)
    original_filename: Mapped[str] = mapped_column(String(512))
    image_hash: Mapped[str] = mapped_column(String(64), index=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    product: Mapped[Product] = relationship(back_populates="images")

class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(160))
    architecture: Mapped[str] = mapped_column(String(160))
    checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    evaluation_metadata: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class SearchHistory(Base):
    __tablename__ = "search_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_image: Mapped[str] = mapped_column(Text)
    result_product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"), nullable=True)
    similarity_score: Mapped[float] = mapped_column(Float)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("models.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
