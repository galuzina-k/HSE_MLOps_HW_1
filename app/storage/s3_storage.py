import os
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.utils.logger import logger


class S3Storage:
    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket_name: str = "mlops-models",
    ):
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        self.access_key = access_key or os.getenv("S3_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("S3_SECRET_KEY", "minioadmin")
        self.bucket_name = bucket_name

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
        )

        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' exists")
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' created")
            except Exception as e:
                logger.warning(f"Could not create bucket: {e}")

    def upload_file(self, file_path: str | Path, object_name: str | None = None):
        file_path = Path(file_path)
        if object_name is None:
            object_name = file_path.name

        try:
            self.s3_client.upload_file(str(file_path), self.bucket_name, object_name)
            logger.info(f"Uploaded {file_path} to S3 as {object_name}")
        except Exception as e:
            logger.error(f"Failed to upload {file_path} to S3: {e}")
            raise

    def download_file(self, object_name: str, file_path: str | Path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.s3_client.download_file(self.bucket_name, object_name, str(file_path))
            logger.info(f"Downloaded {object_name} from S3 to {file_path}")
        except Exception as e:
            logger.error(f"Failed to download {object_name} from S3: {e}")
            raise

    def upload_bytes(self, data: bytes, object_name: str):
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=object_name, Body=data)
            logger.info(f"Uploaded bytes to S3 as {object_name}")
        except Exception as e:
            logger.error(f"Failed to upload bytes to S3: {e}")
            raise

    def download_bytes(self, object_name: str) -> bytes:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_name)
            return response["Body"].read()
        except Exception as e:
            logger.error(f"Failed to download {object_name} from S3: {e}")
            raise

    def delete_file(self, object_name: str):
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            logger.info(f"Deleted {object_name} from S3")
        except Exception as e:
            logger.error(f"Failed to delete {object_name} from S3: {e}")
            raise

    def list_objects(self, prefix: str = "") -> list[str]:
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if "Contents" in response:
                return [obj["Key"] for obj in response["Contents"]]
            return []
        except Exception as e:
            logger.error(f"Failed to list objects from S3: {e}")
            raise

    def object_exists(self, object_name: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except ClientError:
            return False

