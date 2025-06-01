"""
MongoDB Service Layer (Refactored)

- Handles MongoDB client and database connection logic.
- Designed for auditability, security, and testability (Stripe/Netflix standards).
- Implements audit logging and custom exceptions for compliance.
- Provides CRUD wrappers for testability and modularity.
"""

import logging
from typing import Any, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

logger = logging.getLogger("mongo_service")

class MongoServiceError(Exception):
    """Custom exception for MongoService errors."""
    pass

mongo_client: AsyncIOMotorClient = AsyncIOMotorClient(settings.MONGODB_URI)
db: AsyncIOMotorDatabase = mongo_client.get_default_database()

def get_collection(name: str):
    """
    Get a MongoDB collection by name.
    Args:
        name (str): Collection name.
    Returns:
        Collection: MongoDB collection object.
    """
    return db[name]

async def insert_document(collection: str, document: Dict[str, Any]) -> str:
    """
    Insert a document into a MongoDB collection.
    Args:
        collection (str): Collection name.
        document (dict): Document to insert.
    Returns:
        str: Inserted document ID.
    Raises:
        MongoServiceError: If insertion fails.
    """
    try:
        logger.info(f"Inserting document into {collection}")
        result = await get_collection(collection).insert_one(document)
        logger.info(f"Inserted document with id: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Mongo insert error: {e}")
        raise MongoServiceError(f"Mongo insert error: {e}")

async def find_document(collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find a single document in a MongoDB collection.
    Args:
        collection (str): Collection name.
        query (dict): Query filter.
    Returns:
        Optional[dict]: Document if found, else None.
    Raises:
        MongoServiceError: If query fails.
    """
    try:
        logger.info(f"Finding document in {collection} with query: {query}")
        doc = await get_collection(collection).find_one(query)
        return doc
    except Exception as e:
        logger.error(f"Mongo find error: {e}")
        raise MongoServiceError(f"Mongo find error: {e}")

# Test block for service sanity (not for production)
if __name__ == "__main__":
    import asyncio
    async def test():
        try:
            doc_id = await insert_document("test_collection", {"foo": "bar"})
            print(f"Inserted: {doc_id}")
            doc = await find_document("test_collection", {"_id": doc_id})
            print(f"Found: {doc}")
        except Exception as e:
            print(f"Error: {e}")
    asyncio.run(test())
