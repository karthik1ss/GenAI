"""
Unit tests for the traffic ETL module
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from project_spati.traffic_etl import (
    UsageMetric,
    TrafficDataProcessor,
    CrossAccountS3Client,
    TrafficDataETL,
    lambda_handler
)


class TestUsageMetric:
    """Test cases for UsageMetric data model"""
    
    def test_usage_metric_creation(self):
        """Test creating a UsageMetric instance"""
        metric = UsageMetric(
            timestamp="2024-01-15T14:00:00Z",
            api_endpoint="/fba/inbound/v0/shipments",
            request_count=1250,
            throttle_count=45,
            error_count=12,
            response_time_p95=850.5,
            builder_id="builder_12345",
            service_name="FBA_Inbound_Service",
            region="us-east-1"
        )
        
        assert metric.timestamp == "2024-01-15T14:00:00Z"
        assert metric.api_endpoint == "/fba/inbound/v0/shipments"
        assert metric.request_count == 1250
        assert metric.throttle_count == 45
        assert metric.error_count == 12
        assert metric.response_time_p95 == 850.5
        assert metric.builder_id == "builder_12345"
        assert metric.service_name == "FBA_Inbound_Service"
        assert metric.region == "us-east-1"
    
    def test_usage_metric_to_dict(self):
        """Test converting UsageMetric to dictionary"""
        metric = UsageMetric(
            timestamp="2024-01-15T14:00:00Z",
            api_endpoint="/fba/inbound/v0/shipments",
            request_count=1250,
            throttle_count=45,
            error_count=12,
            response_time_p95=850.5,
            builder_id="builder_12345",
            service_name="FBA_Inbound_Service",
            region="us-east-1"
        )
        
        result = metric.to_dict()
        expected = {
            "timestamp": "2024-01-15T14:00:00Z",
            "api_endpoint": "/fba/inbound/v0/shipments",
            "request_count": 1250,
            "throttle_count": 45,
            "error_count": 12,
            "response_time_p95": 850.5,
            "builder_id": "builder_12345",
            "service_name": "FBA_Inbound_Service",
            "region": "us-east-1"
        }
        
        assert result == expected


class TestTrafficDataProcessor:
    """Test cases for TrafficDataProcessor"""
    
    def test_parse_usage_metrics_valid_data(self):
        """Test parsing valid usage metrics data"""
        processor = TrafficDataProcessor()
        raw_data = '''{"timestamp": "2024-01-15T14:00:00Z", "api_endpoint": "/fba/inbound/v0/shipments", "request_count": 1250, "throttle_count": 45, "error_count": 12, "response_time_p95": 850.5, "builder_id": "builder_12345", "service_name": "FBA_Inbound_Service", "region": "us-east-1"}
{"timestamp": "2024-01-15T14:01:00Z", "api_endpoint": "/fba/outbound/v0/orders", "request_count": 800, "throttle_count": 20, "error_count": 5, "response_time_p95": 650.0, "builder_id": "builder_67890", "service_name": "FBA_Outbound_Service", "region": "us-west-2"}'''
        
        metrics = processor.parse_usage_metrics(raw_data)
        
        assert len(metrics) == 2
        assert processor.processed_count == 2
        assert processor.error_count == 0
        
        # Check first metric
        assert metrics[0].timestamp == "2024-01-15T14:00:00Z"
        assert metrics[0].api_endpoint == "/fba/inbound/v0/shipments"
        assert metrics[0].request_count == 1250
        
        # Check second metric
        assert metrics[1].timestamp == "2024-01-15T14:01:00Z"
        assert metrics[1].api_endpoint == "/fba/outbound/v0/orders"
        assert metrics[1].request_count == 800
    
    def test_parse_usage_metrics_invalid_json(self):
        """Test parsing data with invalid JSON lines"""
        processor = TrafficDataProcessor()
        raw_data = '''{"timestamp": "2024-01-15T14:00:00Z", "api_endpoint": "/fba/inbound/v0/shipments", "request_count": 1250}
invalid json line
{"timestamp": "2024-01-15T14:01:00Z", "api_endpoint": "/fba/outbound/v0/orders", "request_count": 800}'''
        
        metrics = processor.parse_usage_metrics(raw_data)
        
        assert len(metrics) == 2  # Only valid lines processed
        assert processor.processed_count == 2
        assert processor.error_count == 1  # One invalid line
    
    def test_parse_usage_metrics_empty_data(self):
        """Test parsing empty data"""
        processor = TrafficDataProcessor()
        raw_data = ""
        
        metrics = processor.parse_usage_metrics(raw_data)
        
        assert len(metrics) == 0
        assert processor.processed_count == 0
        assert processor.error_count == 0


@patch('project_spati.traffic_etl.boto3')
class TestCrossAccountS3Client:
    """Test cases for CrossAccountS3Client"""
    
    def test_assume_role_success(self, mock_boto3):
        """Test successful role assumption"""
        mock_sts = Mock()
        mock_boto3.client.return_value = mock_sts
        
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "test_key",
                "SecretAccessKey": "test_secret",
                "SessionToken": "test_token"
            }
        }
        
        client = CrossAccountS3Client("arn:aws:iam::123456789012:role/TestRole")
        s3_client = client._get_assumed_client()
        
        mock_sts.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::123456789012:role/TestRole",
            RoleSessionName="ProjectSPATI-TrafficETL"
        )
        assert s3_client is not None
    
    def test_list_objects_success(self, mock_boto3):
        """Test successful object listing"""
        mock_s3 = Mock()
        mock_boto3.client.return_value = mock_s3
        
        mock_s3.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "test_key",
                "SecretAccessKey": "test_secret",
                "SessionToken": "test_token"
            }
        }
        
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "test/file1.json"},
                {"Key": "test/file2.json"}
            ]
        }
        
        client = CrossAccountS3Client("arn:aws:iam::123456789012:role/TestRole")
        objects = client.list_objects("test-bucket", "test/")
        
        assert len(objects) == 2
        assert objects[0]["Key"] == "test/file1.json"
        assert objects[1]["Key"] == "test/file2.json"


@patch('project_spati.traffic_etl.s3_client')
@patch('project_spati.traffic_etl.CrossAccountS3Client')
class TestTrafficDataETL:
    """Test cases for TrafficDataETL"""
    
    def test_process_hourly_metrics_success(self, mock_cross_account_client_class, mock_s3_client):
        """Test successful hourly metrics processing"""
        # Setup mocks
        mock_cross_account_client = Mock()
        mock_cross_account_client_class.return_value = mock_cross_account_client
        
        mock_cross_account_client.list_objects.return_value = [
            {"Key": "year=2024/month=01/day=15/hour=14/metrics.json"}
        ]
        
        mock_cross_account_client.get_object.return_value = b'{"timestamp": "2024-01-15T14:00:00Z", "api_endpoint": "/fba/inbound/v0/shipments", "request_count": 1250, "throttle_count": 45, "error_count": 12, "response_time_p95": 850.5, "builder_id": "builder_12345", "service_name": "FBA_Inbound_Service", "region": "us-east-1"}'
        
        # Create ETL instance
        with patch.dict('os.environ', {
            'DEST_BUCKET': 'test-dest-bucket',
            'SOURCE_BUCKET': 'test-source-bucket',
            'CROSS_ACCOUNT_ROLE_ARN': 'arn:aws:iam::123456789012:role/TestRole'
        }):
            etl = TrafficDataETL()
            
            target_hour = datetime(2024, 1, 15, 14, 0, 0)
            result = etl.process_hourly_metrics(target_hour)
            
            assert result["status"] == "success"
            assert result["processed_files"] == 1
            assert result["processed_metrics"] == 1
            assert result["processing_errors"] == 0
            assert result["target_hour"] == "2024-01-15T14:00:00"
    
    def test_process_hourly_metrics_no_files(self, mock_cross_account_client_class, mock_s3_client):
        """Test processing when no files are found"""
        # Setup mocks
        mock_cross_account_client = Mock()
        mock_cross_account_client_class.return_value = mock_cross_account_client
        
        mock_cross_account_client.list_objects.return_value = []
        
        # Create ETL instance
        with patch.dict('os.environ', {
            'DEST_BUCKET': 'test-dest-bucket',
            'SOURCE_BUCKET': 'test-source-bucket',
            'CROSS_ACCOUNT_ROLE_ARN': 'arn:aws:iam::123456789012:role/TestRole'
        }):
            etl = TrafficDataETL()
            
            target_hour = datetime(2024, 1, 15, 14, 0, 0)
            result = etl.process_hourly_metrics(target_hour)
            
            assert result["status"] == "success"
            assert result["processed_files"] == 0
            assert result["processed_metrics"] == 0


@patch('project_spati.traffic_etl.TrafficDataETL')
class TestLambdaHandler:
    """Test cases for lambda_handler function"""
    
    def test_lambda_handler_success(self, mock_etl_class):
        """Test successful lambda handler execution"""
        mock_etl = Mock()
        mock_etl_class.return_value = mock_etl
        
        mock_etl.process_hourly_metrics.return_value = {
            "status": "success",
            "processed_files": 5,
            "processed_metrics": 100,
            "processing_errors": 0
        }
        
        event = {}
        context = Mock()
        
        result = lambda_handler(event, context)
        
        assert result["status"] == "success"
        assert result["processed_files"] == 5
        assert result["processed_metrics"] == 100
        assert result["processing_errors"] == 0
    
    def test_lambda_handler_with_target_hour(self, mock_etl_class):
        """Test lambda handler with specific target hour"""
        mock_etl = Mock()
        mock_etl_class.return_value = mock_etl
        
        mock_etl.process_hourly_metrics.return_value = {
            "status": "success",
            "processed_files": 3,
            "processed_metrics": 50,
            "processing_errors": 1
        }
        
        event = {"target_hour": "2024-01-15T14:00:00"}
        context = Mock()
        
        result = lambda_handler(event, context)
        
        # Verify that process_hourly_metrics was called with the correct datetime
        mock_etl.process_hourly_metrics.assert_called_once()
        call_args = mock_etl.process_hourly_metrics.call_args[0]
        assert call_args[0] == datetime(2024, 1, 15, 14, 0, 0)
        
        assert result["status"] == "success"
    
    def test_lambda_handler_error(self, mock_etl_class):
        """Test lambda handler error handling"""
        mock_etl = Mock()
        mock_etl_class.return_value = mock_etl
        
        mock_etl.process_hourly_metrics.side_effect = Exception("Test error")
        
        event = {}
        context = Mock()
        
        result = lambda_handler(event, context)
        
        assert result["status"] == "error"
        assert "Test error" in result["error_message"]