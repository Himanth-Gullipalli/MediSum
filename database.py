"""
database.py – MongoDB operations for MediSum.

Handles patient records and summary storage.
Gracefully degrades when MongoDB is unavailable.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get_mongo_config() -> tuple[str, str]:
    """Get MongoDB URI and DB name from Streamlit secrets or env."""
    try:
        import streamlit as st
        uri = st.secrets.get("MONGO_URI", os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
        db_name = st.secrets.get("MONGO_DB_NAME", os.getenv("MONGO_DB_NAME", "medisum"))
    except Exception:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("MONGO_DB_NAME", "medisum")
    return uri, db_name


class MediSumDB:
    """Wrapper around pymongo for MediSum patient & summary operations."""

    def __init__(self):
        self._client: Optional[MongoClient] = None
        self._db = None
        self._connected = False
        self._connect()

    # ── Connection ────────────────────────────────────────────

    def _connect(self) -> None:
        uri, db_name = _get_mongo_config()
        try:
            self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Trigger connection check
            self._client.admin.command("ping")
            self._db = self._client[db_name]
            self._connected = True
            self._ensure_indexes()
            logger.info("MongoDB connected: %s / %s", uri, db_name)
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            logger.warning("MongoDB unavailable – running in demo mode. Error: %s", exc)
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _ensure_indexes(self) -> None:
        """Create indexes for efficient queries."""
        if not self._connected:
            return
        self._db.patients.create_index("name")
        self._db.summaries.create_index([("patient_id", 1), ("created_at", DESCENDING)])

    # ── Patient Operations ────────────────────────────────────

    def create_patient(
        self,
        name: str,
        age: int,
        gender: str,
        contact: str = "",
        blood_group: str = "",
        notes: str = "",
    ) -> Optional[str]:
        """
        Insert a new patient record.
        Returns the string patient_id on success, None on failure.
        """
        if not self._connected:
            return None
        doc = {
            "name": name.strip(),
            "age": int(age),
            "gender": gender,
            "contact": contact.strip(),
            "blood_group": blood_group.strip(),
            "notes": notes.strip(),
            "created_at": datetime.now(timezone.utc),
        }
        result = self._db.patients.insert_one(doc)
        return str(result.inserted_id)

    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Retrieve a single patient by ObjectId string. Returns None if not found."""
        if not self._connected:
            return None
        try:
            doc = self._db.patients.find_one({"_id": ObjectId(patient_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception as exc:
            logger.error("get_patient error: %s", exc)
            return None

    def search_patients(self, query: str) -> list[dict]:
        """
        Case-insensitive search on patient name or contact.
        Returns list of patient dicts.
        """
        if not self._connected:
            return []
        pattern = {"$regex": query, "$options": "i"}
        cursor = self._db.patients.find(
            {"$or": [{"name": pattern}, {"contact": pattern}]}
        ).sort("name", 1).limit(50)
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    def list_all_patients(self) -> list[dict]:
        """Return all patients sorted alphabetically by name."""
        if not self._connected:
            return []
        results = []
        for doc in self._db.patients.find().sort("name", 1):
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    def update_patient(self, patient_id: str, updates: dict) -> bool:
        """Update patient fields. Returns True on success."""
        if not self._connected:
            return False
        try:
            result = self._db.patients.update_one(
                {"_id": ObjectId(patient_id)},
                {"$set": {**updates, "updated_at": datetime.now(timezone.utc)}},
            )
            return result.modified_count > 0
        except Exception as exc:
            logger.error("update_patient error: %s", exc)
            return False

    def delete_patient(self, patient_id: str) -> bool:
        """Delete a patient and all their summaries. Returns True on success."""
        if not self._connected:
            return False
        try:
            self._db.patients.delete_one({"_id": ObjectId(patient_id)})
            self._db.summaries.delete_many({"patient_id": patient_id})
            return True
        except Exception as exc:
            logger.error("delete_patient error: %s", exc)
            return False

    # ── Summary Operations ────────────────────────────────────

    def save_summary(
        self,
        patient_id: str,
        summary_text: str,
        report_types: list[str],
        pdf_bytes: Optional[bytes] = None,
    ) -> Optional[str]:
        """
        Persist a generated summary for a patient.
        Returns the summary_id string on success.
        """
        if not self._connected:
            return None
        doc = {
            "patient_id": patient_id,
            "summary_text": summary_text,
            "report_types": report_types,
            "pdf_bytes": pdf_bytes,
            "created_at": datetime.now(timezone.utc),
        }
        result = self._db.summaries.insert_one(doc)
        return str(result.inserted_id)

    def get_summaries(self, patient_id: str) -> list[dict]:
        """
        Retrieve all summaries for a patient, most recent first.
        pdf_bytes is excluded from results for performance (fetch separately).
        """
        if not self._connected:
            return []
        cursor = self._db.summaries.find(
            {"patient_id": patient_id},
            {"pdf_bytes": 0},  # Exclude large binary from listing
        ).sort("created_at", DESCENDING)
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    def get_summary_pdf(self, summary_id: str) -> Optional[bytes]:
        """Fetch stored PDF bytes for a summary."""
        if not self._connected:
            return None
        try:
            doc = self._db.summaries.find_one(
                {"_id": ObjectId(summary_id)},
                {"pdf_bytes": 1},
            )
            return doc.get("pdf_bytes") if doc else None
        except Exception:
            return None

    def get_stats(self) -> dict:
        """Return dashboard statistics."""
        if not self._connected:
            return {"patients": 0, "summaries": 0, "connected": False}
        return {
            "patients": self._db.patients.count_documents({}),
            "summaries": self._db.summaries.count_documents({}),
            "connected": True,
        }
