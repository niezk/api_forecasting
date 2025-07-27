from firebase_config import firebase_db
from google.cloud.firestore_v1.base_query import FieldFilter

def get_firestore_docs():
    docs = (
        firebase_db.collection("forecasting_data")
        .where(filter=FieldFilter("type", "==", "real_data"))
        .stream()
    )
    return [doc.to_dict() | {"id": doc.id} for doc in docs]