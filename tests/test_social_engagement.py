"""
Unit tests for Social Engagement (SE) construct features.

Tests cover communication frequency, contact diversity, and initiation rate
feature extraction with various data quality scenarios and edge cases.
"""

import pytest
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List
from psyconstruct.features.social_engagement import (
    SocialEngagementFeatures,
    CommunicationFrequencyConfig,
    ContactDiversityConfig,
    InitiationRateConfig
)


class TestCommunicationFrequencyConfig:
    """Test CommunicationFrequencyConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CommunicationFrequencyConfig()
        
        assert config.analysis_window_days == 7
        assert config.min_communications_per_day == 1
        assert config.include_incoming == False
        assert config.include_outgoing == True
        assert config.min_days_with_data == 5
        assert config.min_data_coverage == 0.6
        assert config.exclude_weekends == False
        assert config.smooth_daily_counts == True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CommunicationFrequencyConfig(
            analysis_window_days=14,
            min_communications_per_day=2,
            include_incoming=True,
            exclude_weekends=True,
            smooth_daily_counts=False
        )
        
        assert config.analysis_window_days == 14
        assert config.min_communications_per_day == 2
        assert config.include_incoming == True
        assert config.exclude_weekends == True
        assert config.smooth_daily_counts == False


class TestContactDiversityConfig:
    """Test ContactDiversityConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ContactDiversityConfig()
        
        assert config.rolling_window_days == 7
        assert config.analysis_window_days == 14
        assert config.min_interactions_per_contact == 1
        assert config.exclude_auto_messages == True
        assert config.contact_grouping == True
        assert config.min_communications_total == 10
        assert config.min_days_with_data == 5
        assert config.normalize_by_frequency == False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ContactDiversityConfig(
            rolling_window_days=14,
            min_interactions_per_contact=3,
            exclude_auto_messages=False,
            normalize_by_frequency=True
        )
        
        assert config.rolling_window_days == 14
        assert config.min_interactions_per_contact == 3
        assert config.exclude_auto_messages == False
        assert config.normalize_by_frequency == True


class TestInitiationRateConfig:
    """Test InitiationRateConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = InitiationRateConfig()
        
        assert config.analysis_window_days == 7
        assert config.min_total_communications == 5
        assert config.handle_zero_division == "return_zero"
        assert config.include_only_bidirectional == False
        assert config.min_days_with_data == 5
        assert config.min_data_coverage == 0.5
        assert config.calculate_by_contact == True
        assert config.exclude_sparse_contacts == True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = InitiationRateConfig(
            analysis_window_days=14,
            min_total_communications=10,
            handle_zero_division="return_nan",
            include_only_bidirectional=True
        )
        
        assert config.analysis_window_days == 14
        assert config.min_total_communications == 10
        assert config.handle_zero_division == "return_nan"
        assert config.include_only_bidirectional == True


class TestCommunicationFrequency:
    """Test communication frequency feature extraction."""
    
    def create_sample_communication_data(self, days: int = 7, communications_per_day: int = 10):
        """Create sample communication data."""
        timestamps = []
        directions = []
        contacts = []
        
        base_time = datetime(2026, 2, 21, 8, 0, 0)
        
        for day in range(days):
            for comm in range(communications_per_day):
                hour = 8 + comm * (16 // communications_per_day)  # Spread across 8 AM - midnight
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                # Mix of incoming and outgoing
                direction = 'outgoing' if comm % 3 != 0 else 'incoming'
                contact = f'contact_{comm % 5}'
                
                timestamps.append(timestamp)
                directions.append(direction)
                contacts.append(contact)
        
        return {
            'timestamp': timestamps,
            'direction': directions,
            'contact': contacts
        }
    
    def test_communication_frequency_basic(self):
        """Test basic communication frequency calculation."""
        config = CommunicationFrequencyConfig(
            min_communications_per_day=1,
            analysis_window_days=7
        )
        features = SocialEngagementFeatures(freq_config=config)
        
        # Create communication data
        comm_data = self.create_sample_communication_data(days=5, communications_per_day=8)
        
        result = features.communication_frequency(comm_data)
        
        # Check result structure
        assert 'communication_frequency' in result
        assert 'weekly_outgoing_count' in result
        assert 'daily_frequency' in result
        assert 'quality_metrics' in result
        assert 'processing_parameters' in result
        assert 'data_summary' in result
        
        # Check values
        assert result['weekly_outgoing_count'] >= 0
        assert len(result['daily_frequency']) > 0
        assert result['communication_frequency']['mean_daily_frequency'] >= 0
        
        # Check daily frequency structure
        for date, freq_data in result['daily_frequency'].items():
            assert 'total_communications' in freq_data
            assert 'outgoing_communications' in freq_data
            assert 'incoming_communications' in freq_data
            assert 'is_active' in freq_data
    
    def test_communication_frequency_insufficient_data(self):
        """Test communication frequency with insufficient data."""
        config = CommunicationFrequencyConfig(min_communications_per_day=5)
        features = SocialEngagementFeatures(freq_config=config)
        
        # Create sparse data
        sparse_data = {
            'timestamp': [datetime.now()],
            'direction': ['outgoing'],
            'contact': ['contact_1']
        }
        
        with pytest.raises(ValueError, match="Insufficient communication data"):
            features.communication_frequency(sparse_data)
    
    def test_communication_frequency_weekend_exclusion(self):
        """Test communication frequency with weekend exclusion."""
        config = CommunicationFrequencyConfig(
            exclude_weekends=True,
            min_communications_per_day=1
        )
        features = SocialEngagementFeatures(freq_config=config)
        
        # Create data spanning weekdays and weekends
        comm_data = {
            'timestamp': [datetime(2026, 2, 21 + i, 12, 0, 0) for i in range(7)],  # Mon-Sun
            'direction': ['outgoing'] * 7,
            'contact': [f'contact_{i}' for i in range(7)]
        }
        
        result = features.communication_frequency(comm_data)
        
        # Should only include weekdays (Mon-Fri)
        assert len(result['daily_frequency']) <= 5
        
        # Check that weekend dates are excluded
        for date_str in result['daily_frequency']:
            date = datetime.fromisoformat(date_str)
            assert date.weekday() < 5  # Monday-Friday
    
    def test_communication_frequency_incoming_inclusion(self):
        """Test communication frequency with incoming inclusion."""
        # Test with only outgoing
        config_outgoing = CommunicationFrequencyConfig(
            include_incoming=False,
            include_outgoing=True,
            min_communications_per_day=1
        )
        features_outgoing = SocialEngagementFeatures(freq_config=config_outgoing)
        
        # Test with both directions
        config_both = CommunicationFrequencyConfig(
            include_incoming=True,
            include_outgoing=True,
            min_communications_per_day=1
        )
        features_both = SocialEngagementFeatures(freq_config=config_both)
        
        comm_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(10)],
            'direction': ['outgoing', 'incoming'] * 5,
            'contact': [f'contact_{i}' for i in range(10)]
        }
        
        result_outgoing = features_outgoing.communication_frequency(comm_data)
        result_both = features_both.communication_frequency(comm_data)
        
        # Outgoing-only should have lower count than both directions
        assert result_outgoing['weekly_outgoing_count'] <= result_both['weekly_outgoing_count']
    
    def test_communication_frequency_smoothing(self):
        """Test communication frequency with smoothing."""
        # Test with smoothing
        config_smooth = CommunicationFrequencyConfig(
            smooth_daily_counts=True,
            min_communications_per_day=1
        )
        features_smooth = SocialEngagementFeatures(freq_config=config_smooth)
        
        # Test without smoothing
        config_no_smooth = CommunicationFrequencyConfig(
            smooth_daily_counts=False,
            min_communications_per_day=1
        )
        features_no_smooth = SocialEngagementFeatures(freq_config=config_no_smooth)
        
        # Create data with varying daily counts
        comm_data = {
            'timestamp': [datetime(2026, 2, 21 + i, 12, 0, 0) for i in range(5)],
            'direction': ['outgoing'] * 5,
            'contact': [f'contact_{i}' for i in range(5)]
        }
        
        result_smooth = features_smooth.communication_frequency(comm_data)
        result_no_smooth = features_no_smooth.communication_frequency(comm_data)
        
        # Both should produce valid results
        assert result_smooth['weekly_outgoing_count'] >= 0
        assert result_no_smooth['weekly_outgoing_count'] >= 0
    
    def test_communication_data_validation(self):
        """Test communication data validation."""
        features = SocialEngagementFeatures()
        
        # Missing required column
        invalid_data = {
            'timestamp': [datetime.now()],
            'direction': ['outgoing']
            # Missing 'contact'
        }
        
        with pytest.raises(ValueError, match="Missing required column: contact"):
            features.communication_frequency(invalid_data)
        
        # Invalid direction
        invalid_direction = {
            'timestamp': [datetime.now()],
            'direction': ['invalid'],
            'contact': ['contact_1']
        }
        
        with pytest.raises(ValueError, match="Invalid direction"):
            features.communication_frequency(invalid_direction)


class TestContactDiversity:
    """Test contact diversity feature extraction."""
    
    def create_diverse_communication_data(self, days: int = 7, contacts_per_day: int = 5):
        """Create communication data with diverse contacts."""
        timestamps = []
        directions = []
        contacts = []
        
        base_time = datetime(2026, 2, 21, 8, 0, 0)
        
        for day in range(days):
            for comm in range(contacts_per_day):
                hour = 8 + comm * (16 // contacts_per_day)
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                direction = 'outgoing' if comm % 2 == 0 else 'incoming'
                contact = f'contact_{day}_{comm % 10}'  # Diverse contacts
                
                timestamps.append(timestamp)
                directions.append(direction)
                contacts.append(contact)
        
        return {
            'timestamp': timestamps,
            'direction': directions,
            'contact': contacts
        }
    
    def test_contact_diversity_basic(self):
        """Test basic contact diversity calculation."""
        config = ContactDiversityConfig(
            min_communications_total=10,
            rolling_window_days=7
        )
        features = SocialEngagementFeatures(diversity_config=config)
        
        # Create diverse communication data
        comm_data = self.create_diverse_communication_data(days=7, contacts_per_day=5)
        
        result = features.contact_diversity(comm_data)
        
        # Check result structure
        assert 'contact_diversity' in result
        assert 'weekly_diversity' in result
        assert 'rolling_diversity' in result
        assert 'contact_statistics' in result
        assert 'quality_metrics' in result
        assert 'processing_parameters' in result
        assert 'data_summary' in result
        
        # Check values
        assert result['weekly_diversity'] >= 0
        assert len(result['rolling_diversity']) > 0
        assert result['contact_diversity']['mean_rolling_diversity'] >= 0
        
        # Check contact statistics
        for contact, stats in result['contact_statistics'].items():
            assert 'total_communications' in stats
            assert 'first_contact' in stats
            assert 'last_contact' in stats
            assert 'communication_span_days' in stats
    
    def test_contact_diversity_insufficient_data(self):
        """Test contact diversity with insufficient data."""
        config = ContactDiversityConfig(min_communications_total=20)
        features = SocialEngagementFeatures(diversity_config=config)
        
        # Create sparse data
        sparse_data = {
            'timestamp': [datetime.now()],
            'direction': ['outgoing'],
            'contact': ['contact_1']
        }
        
        with pytest.raises(ValueError, match="Insufficient communication data"):
            features.contact_diversity(sparse_data)
    
    def test_contact_diversity_minimum_interactions_filter(self):
        """Test contact diversity with minimum interactions filter."""
        config = ContactDiversityConfig(
            min_interactions_per_contact=3,
            min_communications_total=10
        )
        features = SocialEngagementFeatures(diversity_config=config)
        
        # Create data with some contacts having few interactions
        comm_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(15)],
            'direction': ['outgoing'] * 15,
            'contact': ['contact_1'] * 5 + ['contact_2'] * 3 + ['contact_3'] * 7  # contact_2 has only 3
        }
        
        result = features.contact_diversity(comm_data)
        
        # Should only include contacts with >= 3 interactions
        assert 'contact_1' in result['contact_statistics']
        assert 'contact_3' in result['contact_statistics']
        # contact_2 has exactly 3, so it should be included
    
    def test_contact_diversity_auto_message_exclusion(self):
        """Test contact diversity with automated message exclusion."""
        config = ContactDiversityConfig(
            exclude_auto_messages=True,
            min_communications_total=5
        )
        features = SocialEngagementFeatures(diversity_config=config)
        
        # Create data with automated messages
        comm_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(8)],
            'direction': ['outgoing'] * 8,
            'contact': ['system_alert', 'contact_1', 'auto_bot', 'contact_2', 'notification_service', 'contact_3', 'contact_1', 'contact_2']
        }
        
        result = features.contact_diversity(comm_data)
        
        # Should exclude automated contacts
        assert 'system_alert' not in result['contact_statistics']
        assert 'auto_bot' not in result['contact_statistics']
        assert 'notification_service' not in result['contact_statistics']
        
        # Should include regular contacts
        assert 'contact_1' in result['contact_statistics']
        assert 'contact_2' in result['contact_statistics']
        assert 'contact_3' in result['contact_statistics']
    
    def test_contact_diversity_rolling_window(self):
        """Test contact diversity rolling window calculation."""
        config = ContactDiversityConfig(
            rolling_window_days=3,
            min_communications_total=5
        )
        features = SocialEngagementFeatures(diversity_config=config)
        
        # Create data with different contacts over time
        comm_data = {
            'timestamp': [datetime(2026, 2, 21 + i, 12, 0, 0) for i in range(7)],  # 7 days
            'direction': ['outgoing'] * 7,
            'contact': ['contact_1', 'contact_2', 'contact_3', 'contact_4', 'contact_5', 'contact_6', 'contact_7']
        }
        
        result = features.contact_diversity(comm_data)
        
        # Should have rolling diversity for each day
        assert len(result['rolling_diversity']) == 7
        
        # Rolling diversity should increase over time (more contacts in window)
        diversity_values = list(result['rolling_diversity'].values())
        for i in range(1, len(diversity_values)):
            # Should be non-decreasing for first 3 days, then stable
            if i < 3:
                assert diversity_values[i] >= diversity_values[i-1]
    
    def test_diversity_stability_calculation(self):
        """Test diversity stability calculation."""
        features = SocialEngagementFeatures()
        
        # Create stable diversity (same contacts each day)
        stable_data = {
            'timestamp': [datetime(2026, 2, 21 + i, 12, 0, 0) for i in range(5)],
            'direction': ['outgoing'] * 5,
            'contact': ['contact_1', 'contact_2', 'contact_1', 'contact_2', 'contact_1']
        }
        
        stable_result = features.contact_diversity(stable_data)
        
        # Create variable diversity (different contacts each day)
        variable_data = {
            'timestamp': [datetime(2026, 2, 21 + i, 12, 0, 0) for i in range(5)],
            'direction': ['outgoing'] * 5,
            'contact': ['contact_1', 'contact_2', 'contact_3', 'contact_4', 'contact_5']
        }
        
        variable_result = features.contact_diversity(variable_data)
        
        # Stable should have higher stability score
        assert stable_result['contact_diversity']['diversity_stability'] >= variable_result['contact_diversity']['diversity_stability']


class TestInitiationRate:
    """Test initiation rate feature extraction."""
    
    def create_balanced_communication_data(self, days: int = 7, communications_per_day: int = 6):
        """Create communication data with balanced incoming/outgoing."""
        timestamps = []
        directions = []
        contacts = []
        
        base_time = datetime(2026, 2, 21, 8, 0, 0)
        
        for day in range(days):
            for comm in range(communications_per_day):
                hour = 8 + comm * (16 // communications_per_day)
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                # Balanced incoming/outgoing
                direction = 'outgoing' if comm % 2 == 0 else 'incoming'
                contact = f'contact_{comm % 4}'
                
                timestamps.append(timestamp)
                directions.append(direction)
                contacts.append(contact)
        
        return {
            'timestamp': timestamps,
            'direction': directions,
            'contact': contacts
        }
    
    def test_initiation_rate_basic(self):
        """Test basic initiation rate calculation."""
        config = InitiationRateConfig(
            min_total_communications=5,
            handle_zero_division="return_zero"
        )
        features = SocialEngagementFeatures(initiation_config=config)
        
        # Create balanced communication data
        comm_data = self.create_balanced_communication_data(days=5, communications_per_day=6)
        
        result = features.initiation_rate(comm_data)
        
        # Check result structure
        assert 'initiation_rate' in result
        assert 'weekly_initiation_rate' in result
        assert 'contact_initiation_rates' in result
        assert 'direction_counts' in result
        assert 'quality_metrics' in result
        assert 'processing_parameters' in result
        assert 'data_summary' in result
        
        # Check values
        assert 0 <= result['weekly_initiation_rate'] <= 1
        assert len(result['contact_initiation_rates']) > 0
        assert result['initiation_rate']['direction_counts']['total'] > 0
        
        # Check contact initiation rates
        for contact, rate_data in result['contact_initiation_rates'].items():
            assert 'initiation_rate' in rate_data
            assert 'outgoing_count' in rate_data
            assert 'incoming_count' in rate_data
            assert 'total_count' in rate_data
            assert 0 <= rate_data['initiation_rate'] <= 1
    
    def test_initiation_rate_insufficient_data(self):
        """Test initiation rate with insufficient data."""
        config = InitiationRateConfig(min_total_communications=10)
        features = SocialEngagementFeatures(initiation_config=config)
        
        # Create sparse data
        sparse_data = {
            'timestamp': [datetime.now()],
            'direction': ['outgoing'],
            'contact': ['contact_1']
        }
        
        with pytest.raises(ValueError, match="Insufficient communication data"):
            features.initiation_rate(sparse_data)
    
    def test_initiation_rate_zero_division_handling(self):
        """Test initiation rate zero division handling."""
        # Test return_zero
        config_zero = InitiationRateConfig(
            min_total_communications=1,
            handle_zero_division="return_zero"
        )
        features_zero = SocialEngagementFeatures(initiation_config=config_zero)
        
        # Test return_nan
        config_nan = InitiationRateConfig(
            min_total_communications=1,
            handle_zero_division="return_nan"
        )
        features_nan = SocialEngagementFeatures(initiation_config=config_nan)
        
        # Create data with no communications (edge case - shouldn't happen with min_total_communications)
        # Instead, create data and manually test the logic
        comm_data = {
            'timestamp': [datetime(2026, 2, 21, 12, 0, 0)],
            'direction': ['outgoing'],
            'contact': ['contact_1']
        }
        
        result_zero = features_zero.initiation_rate(comm_data)
        result_nan = features_nan.initiation_rate(comm_data)
        
        # Both should handle the case properly
        assert result_zero['weekly_initiation_rate'] == 1.0  # 1 outgoing / 1 total
        assert result_nan['weekly_initiation_rate'] == 1.0   # 1 outgoing / 1 total
    
    def test_initiation_rate_bidirectional_filter(self):
        """Test initiation rate with bidirectional contact filter."""
        # Test without filter
        config_no_filter = InitiationRateConfig(
            include_only_bidirectional=False,
            min_total_communications=5
        )
        features_no_filter = SocialEngagementFeatures(initiation_config=config_no_filter)
        
        # Test with filter
        config_filter = InitiationRateConfig(
            include_only_bidirectional=True,
            min_total_communications=5
        )
        features_filter = SocialEngagementFeatures(initiation_config=config_filter)
        
        # Create data with some unidirectional contacts
        comm_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(10)],
            'direction': ['outgoing', 'outgoing', 'incoming', 'outgoing', 'incoming'] * 2,
            'contact': ['contact_1', 'contact_1', 'contact_1', 'contact_2', 'contact_2'] * 2
        }
        
        result_no_filter = features_no_filter.initiation_rate(comm_data)
        result_filter = features_filter.initiation_rate(comm_data)
        
        # Filter should include fewer contacts (only bidirectional ones)
        assert len(result_filter['contact_initiation_rates']) <= len(result_no_filter['contact_initiation_rates'])
    
    def test_initiation_rate_sparse_contact_exclusion(self):
        """Test initiation rate with sparse contact exclusion."""
        config = InitiationRateConfig(
            exclude_sparse_contacts=True,
            min_total_communications=5
        )
        features = SocialEngagementFeatures(initiation_config=config)
        
        # Create data with some sparse contacts
        comm_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(10)],
            'direction': ['outgoing'] * 10,
            'contact': ['contact_1'] * 6 + ['contact_2'] * 2 + ['contact_3'] * 2  # contact_2 and contact_3 are sparse
        }
        
        result = features.initiation_rate(comm_data)
        
        # Should exclude sparse contacts (less than 3 interactions)
        assert 'contact_1' in result['contact_initiation_rates']
        # contact_2 and contact_3 have only 2 interactions each, so should be excluded
    
    def test_initiation_rate_calculation_accuracy(self):
        """Test initiation rate calculation accuracy."""
        features = SocialEngagementFeatures()
        
        # Create data with known ratios
        comm_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(10)],
            'direction': ['outgoing'] * 7 + ['incoming'] * 3,  # 70% outgoing
            'contact': [f'contact_{i}' for i in range(10)]
        }
        
        result = features.initiation_rate(comm_data)
        
        # Should calculate 7/10 = 0.7
        assert abs(result['weekly_initiation_rate'] - 0.7) < 0.01
        
        # Check direction counts
        assert result['initiation_rate']['direction_counts']['outgoing'] == 7
        assert result['initiation_rate']['direction_counts']['incoming'] == 3
        assert result['initiation_rate']['direction_counts']['total'] == 10
    
    def test_proactive_contacts_calculation(self):
        """Test proactive contacts calculation."""
        features = SocialEngagementFeatures()
        
        # Create data with mixed initiation patterns
        comm_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(12)],
            'direction': ['outgoing', 'outgoing', 'incoming'] * 4,  # contact_1: 66% proactive
            'contact': ['contact_1'] * 6 + ['contact_2'] * 6  # contact_2: 66% proactive
        }
        
        result = features.initiation_rate(comm_data)
        
        # Both contacts should be proactive (>50% outgoing)
        assert result['initiation_rate']['proactive_contacts'] == 2
        
        # Check individual contact rates
        for contact_data in result['contact_initiation_rates'].values():
            assert contact_data['initiation_rate'] == 0.6666666666666666  # 2/3
    
    def test_direction_balance_calculation(self):
        """Test direction balance calculation."""
        features = SocialEngagementFeatures()
        
        # Perfectly balanced
        balanced_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(10)],
            'direction': ['outgoing'] * 5 + ['incoming'] * 5,
            'contact': [f'contact_{i}' for i in range(10)]
        }
        
        balanced_result = features.initiation_rate(balanced_data)
        assert balanced_result['initiation_rate']['direction_balance'] == 1.0
        
        # Completely unbalanced
        unbalanced_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(10)],
            'direction': ['outgoing'] * 10,
            'contact': [f'contact_{i}' for i in range(10)]
        }
        
        unbalanced_result = features.initiation_rate(unbalanced_data)
        assert unbalanced_result['initiation_rate']['direction_balance'] == 0.0
