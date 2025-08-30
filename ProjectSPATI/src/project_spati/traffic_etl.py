"""
Traffic Data ETL Lambda Function

This module handles the extraction, transformation, and loading of Kopilwuak usage metrics
from S3 for the SP-API Traffic Intelligence system.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import boto3
from aws_lambda_powertools import Logger, Metrics
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext
import gzip
from dataclasses import dataclass, asdict

logger = Logger(service="ProjectSPATI-TrafficETL")
metrics = Metrics(namespace="ProjectSPATI/TrafficETL")

# Environment variables
SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET", "")
DEST_BUCKET = os.environ.get("DEST_BUCKET", "")
CROSS_ACCOUNT_ROLE_ARN = os.environ.get("CROSS_ACCOUNT_ROLE_ARN", "")
AWS_REGION = os.environ["AWS_REGION"]

# Initialize AWS clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
sts_client = boto3.client("sts", region_name=AWS_REGION)


@dataclass
class UsageMetric:
    """Data model for SP-API usage metrics"""
    timestamp: str
    api_endpoint: str
    request_count: int
    throttle_count: int
    error_count: int
    response_time_p95: float
    builder_id: str
    service_name: str
    region: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrossAccountS3Client:
    """Client for accessing cross-account S3 resources"""
    
    def __init__(self, role_arn: str):
        self.role_arn = role_arn
        self._assumed_client = None
    
    def _get_assumed_client(self):
        """Get S3 client with assumed cross-account role"""
        if self._assumed_client is None:
            try:
                response = sts_client.assume_role(
                    RoleArn=self.role_arn,
                    RoleSessionName="ProjectSPATI-TrafficETL"
                )
                credentials = response["Credentials"]
                
                self._assumed_client = boto3.client(
                    "s3",
                    region_name=AWS_REGION,
                    aws_access_key_id=credentials["AccessKeyId"],
                    aws_secret_access_key=credentials["SecretAccessKey"],
                    aws_session_token=credentials["SessionToken"]
                )
                logger.info("Successfully assumed cross-account role")
            except Exception as e:
                logger.error(f"Failed to assume cross-account role: {str(e)}")
                raise
        
        return self._assumed_client
    
    def list_objects(self, bucket: str, prefix: str) -> List[Dict]:
        """List objects in cross-account S3 bucket"""
        client = self._get_assumed_client()
        try:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return response.get("Contents", [])
        except Exception as e:
            logger.error(f"Failed to list objects in {bucket}/{prefix}: {str(e)}")
            raise
    
    def get_object(self, bucket: str, key: str) -> bytes:
        """Get object from cross-account S3 bucket"""
        client = self._get_assumed_client()
        try:
            response = client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            logger.error(f"Failed to get object {bucket}/{key}: {str(e)}")
            raise


class TrafficDataProcessor:
    """Processes raw Kopilwuak usage metrics into structured format"""
    
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
    
    def parse_usage_metrics(self, raw_data: str) -> List[UsageMetric]:
        """Parse raw usage metrics data into structured format"""
        metrics_list = []
        
        try:
            # Assuming the raw data is JSON lines format
            for line in raw_data.strip().split('\n'):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    metric = self._transform_raw_metric(data)
                    if metric:
                        metrics_list.append(metric)
                        self.processed_count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON line: {line[:100]}... Error: {str(e)}")
                    self.error_count += 1
                except Exception as e:
                    logger.warning(f"Failed to transform metric: {str(e)}")
                    self.error_count += 1
                    
        except Exception as e:
            logger.error(f"Failed to parse usage metrics: {str(e)}")
            raise
        
        return metrics_list
    
    def _transform_raw_metric(self, raw_data: Dict) -> Optional[UsageMetric]:
        """Transform raw metric data into UsageMetric object"""
        try:
            # Map raw data fields to our structured format
            # This mapping will need to be adjusted based on actual Kopilwuak data format
            return UsageMetric(
                timestamp=raw_data.get("timestamp", ""),
                api_endpoint=raw_data.get("api_endpoint", ""),
                request_count=int(raw_data.get("request_count", 0)),
                throttle_count=int(raw_data.get("throttle_count", 0)),
                error_count=int(raw_data.get("error_count", 0)),
                response_time_p95=float(raw_data.get("response_time_p95", 0.0)),
                builder_id=raw_data.get("builder_id", ""),
                service_name=raw_data.get("service_name", ""),
                region=raw_data.get("region", AWS_REGION)
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to transform raw metric: {str(e)}")
            return None


class TrafficDataETL:
    """Main ETL orchestrator for traffic data processing"""
    
    def __init__(self):
        self.cross_account_client = CrossAccountS3Client(CROSS_ACCOUNT_ROLE_ARN) if CROSS_ACCOUNT_ROLE_ARN else None
        self.processor = TrafficDataProcessor()
    
    def process_hourly_metrics(self, target_hour: Optional[datetime] = None) -> Dict[str, Any]:
        """Process hourly usage metrics for a specific hour"""
        if target_hour is None:
            # Default to previous hour
            target_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        
        logger.info(f"Processing metrics for hour: {target_hour.isoformat()}")
        
        # Generate S3 key pattern for the target hour
        # Assuming format: year=YYYY/month=MM/day=DD/hour=HH/
        s3_prefix = f"year={target_hour.year}/month={target_hour.month:02d}/day={target_hour.day:02d}/hour={target_hour.hour:02d}/"
        
        try:
            # List files for the target hour
            if self.cross_account_client and SOURCE_BUCKET:
                objects = self.cross_account_client.list_objects(SOURCE_BUCKET, s3_prefix)
            else:
                # For testing without cross-account access
                response = s3_client.list_objects_v2(Bucket=DEST_BUCKET, Prefix=f"test-data/{s3_prefix}")
                objects = response.get("Contents", [])
            
            logger.info(f"Found {len(objects)} files to process")
            
            all_metrics = []
            for obj in objects:
                key = obj["Key"]
                logger.info(f"Processing file: {key}")
                
                # Download and process file
                if self.cross_account_client and SOURCE_BUCKET:
                    raw_data = self.cross_account_client.get_object(SOURCE_BUCKET, key)
                else:
                    # For testing
                    response = s3_client.get_object(Bucket=DEST_BUCKET, Key=key)
                    raw_data = response["Body"].read()
                
                # Handle gzipped files
                if key.endswith('.gz'):
                    raw_data = gzip.decompress(raw_data)
                
                # Parse metrics
                content = raw_data.decode('utf-8')
                metrics_batch = self.processor.parse_usage_metrics(content)
                all_metrics.extend(metrics_batch)
            
            # Store processed metrics
            if all_metrics:
                self._store_processed_metrics(all_metrics, target_hour)
            
            # Record metrics
            metrics.add_metric(name="ProcessedFiles", unit=MetricUnit.Count, value=len(objects))
            metrics.add_metric(name="ProcessedMetrics", unit=MetricUnit.Count, value=len(all_metrics))
            metrics.add_metric(name="ProcessingErrors", unit=MetricUnit.Count, value=self.processor.error_count)
            
            return {
                "status": "success",
                "processed_files": len(objects),
                "processed_metrics": len(all_metrics),
                "processing_errors": self.processor.error_count,
                "target_hour": target_hour.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process hourly metrics: {str(e)}")
            metrics.add_metric(name="ProcessingFailures", unit=MetricUnit.Count, value=1)
            raise
    
    def _store_processed_metrics(self, metrics_list: List[UsageMetric], target_hour: datetime):
        """Store processed metrics in destination S3 bucket"""
        try:
            # Create output key
            output_key = f"traffic-intelligence/processed/year={target_hour.year}/month={target_hour.month:02d}/day={target_hour.day:02d}/hour={target_hour.hour:02d}/metrics.json"
            
            # Convert metrics to JSON
            output_data = {
                "processing_timestamp": datetime.utcnow().isoformat(),
                "target_hour": target_hour.isoformat(),
                "metrics_count": len(metrics_list),
                "metrics": [metric.to_dict() for metric in metrics_list]
            }
            
            # Upload to S3
            s3_client.put_object(
                Bucket=DEST_BUCKET,
                Key=output_key,
                Body=json.dumps(output_data, indent=2),
                ContentType="application/json"
            )
            
            logger.info(f"Stored {len(metrics_list)} processed metrics to {output_key}")
            
        except Exception as e:
            logger.error(f"Failed to store processed metrics: {str(e)}")
            raise


@metrics.log_metrics
def lambda_handler(event: Dict[str, Any], context: LambdaContext) -> Dict[str, Any]:
    """Lambda handler for traffic data ETL processing"""
    logger.info("Starting traffic data ETL processing", extra={"event": event})
    
    try:
        etl = TrafficDataETL()
        
        # Check if specific hour is requested
        target_hour = None
        if "target_hour" in event:
            target_hour = datetime.fromisoformat(event["target_hour"])
        
        result = etl.process_hourly_metrics(target_hour)
        
        logger.info("Traffic data ETL processing completed successfully", extra={"result": result})
        return result
        
    except Exception as e:
        logger.error(f"Traffic data ETL processing failed: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e)
        }