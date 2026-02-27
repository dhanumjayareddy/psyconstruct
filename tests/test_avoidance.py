"""
Unit tests for Avoidance (AV) construct features.

Tests cover home confinement, communication gaps, and movement radius
feature extraction with various data quality scenarios and edge cases.
"""

import pytest
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List
from psyconstruct.features.avoidance import (
    AvoidanceFeatures,
    HomeConfinementConfig,
    CommunicationGapsConfig,
    MovementRadiusConfig
)


class TestHomeConfinementConfig:
    """Test HomeConfinementConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HomeConfinementConfig()
        
        assert config.home_radius_meters == 50.0
        assert config.min_night_points == 10
        assert config.night_start_hour == 22
        assert config.night_end_hour == 6
        assert config.analysis_window_days == 7
        assert config.min_gps_points == 100
        assert config.min_data_coverage == 0.6
        assert config.max_gap_hours == 4.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = HomeConfinementConfig(
            home_radius_meters=100.0,
            min_night_points=15,
            night_start_hour=23,
            night_end_hour=5,
            analysis_window_days=14
        )
        
        assert config.home_radius_meters == 100.0
        assert config.min_night_points == 15
        assert config.night_start_hour == 23
        assert config.night_end_hour == 5
        assert config.analysis_window_days == 14


class TestCommunicationGapsConfig:
    """Test CommunicationGapsConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CommunicationGapsConfig()
        
        assert config.analysis_window_days == 7
        assert config.min_communications == 3
        assert config.max_gap_hours == 24.0
        assert config.min_gap_duration_minutes == 30.0
        assert config.min_days_with_data == 5
        assert config.min_data_coverage == 0.5
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CommunicationGapsConfig(
            analysis_window_days=14,
            min_communications=5,
            min_gap_duration_minutes=60.0,
            max_gap_hours=12.0
        )
        
        assert config.analysis_window_days == 14
        assert config.min_communications == 5
        assert config.min_gap_duration_minutes == 60.0
        assert config.max_gap_hours == 12.0


class TestMovementRadiusConfig:
    """Test MovementRadiusConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MovementRadiusConfig()
        
        assert config.analysis_window_days == 7
        assert config.min_gps_points == 50
        assert config.use_haversine == True
        assert config.outlier_threshold_std == 3.0
        assert config.min_data_coverage == 0.6
        assert config.coordinate_precision == 1e-6
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MovementRadiusConfig(
            analysis_window_days=14,
            use_haversine=False,
            outlier_threshold_std=2.5
        )
        
        assert config.analysis_window_days == 14
        assert config.use_haversine == False
        assert config.outlier_threshold_std == 2.5


class TestHomeConfinement:
    """Test home confinement feature extraction."""
    
    def create_sample_gps_data(self, days: int = 7, points_per_day: int = 24, 
                              home_lat: float = 40.7128, home_lon: float = -74.0060):
        """Create sample GPS data with home and away points."""
        timestamps = []
        latitudes = []
        longitudes = []
        
        base_time = datetime(2026, 2, 21, 0, 0, 0)
        
        for day in range(days):
            for hour in range(points_per_day):
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                # Create pattern: home at night, away during day
                if hour >= 22 or hour < 6:  # Nighttime - home
                    lat = home_lat + (hash(str(day * 100 + hour)) % 100 - 50) * 0.0001
                    lon = home_lon + (hash(str(day * 200 + hour)) % 100 - 50) * 0.0001
                else:  # Daytime - various locations
                    lat = home_lat + (hash(str(day * 300 + hour)) % 200 - 100) * 0.001
                    lon = home_lon + (hash(str(day * 400 + hour)) % 200 - 100) * 0.001
                
                timestamps.append(timestamp)
                latitudes.append(lat)
                longitudes.append(lon)
        
        return {
            'timestamp': timestamps,
            'latitude': latitudes,
            'longitude': longitudes
        }
    
    def test_home_confinement_basic(self):
        """Test basic home confinement calculation."""
        config = HomeConfinementConfig(
            min_night_points=5,
            min_gps_points=20,
            home_radius_meters=100.0
        )
        features = AvoidanceFeatures(home_config=config)
        
        # Create GPS data with clear home pattern
        gps_data = self.create_sample_gps_data(days=5, points_per_day=24)
        
        result = features.home_confinement(gps_data)
        
        # Check result structure
        assert 'home_confinement' in result
        assert 'weekly_confinement_percentage' in result
        assert 'home_location' in result
        assert 'quality_metrics' in result
        assert 'processing_parameters' in result
        assert 'data_summary' in result
        
        # Check values
        assert 0 <= result['weekly_confinement_percentage'] <= 100
        assert result['home_location']['detected'] == True
        assert 'latitude' in result['home_location']
        assert 'longitude' in result['home_location']
        
        # Check home confinement metrics
        confinement = result['home_confinement']
        assert 'confinement_percentage' in confinement
        assert 'points_within_home' in confinement
        assert 'total_points' in confinement
        assert confinement['home_detected'] == True
    
    def test_home_confinement_insufficient_data(self):
        """Test home confinement with insufficient data."""
        config = HomeConfinementConfig(min_gps_points=100)
        features = AvoidanceFeatures(home_config=config)
        
        # Create sparse data
        sparse_data = {
            'timestamp': [datetime.now()],
            'latitude': [40.7128],
            'longitude': [-74.0060]
        }
        
        with pytest.raises(ValueError, match="Insufficient GPS points"):
            features.home_confinement(sparse_data)
    
    def test_home_confinement_no_nighttime_data(self):
        """Test home confinement with insufficient nighttime data."""
        config = HomeConfinementConfig(
            min_night_points=20,
            min_gps_points=10,
            home_radius_meters=100.0
        )
        features = AvoidanceFeatures(home_config=config)
        
        # Create data with only daytime points
        daytime_data = {
            'timestamp': [datetime(2026, 2, 21, 12 + i, 0, 0) for i in range(10)],
            'latitude': [40.7128] * 10,
            'longitude': [-74.0060] * 10
        }
        
        result = features.home_confinement(daytime_data)
        
        # Should not detect home due to insufficient nighttime data
        assert result['home_location']['detected'] == False
        assert result['weekly_confinement_percentage'] == 0.0
    
    def test_home_confinement_custom_config(self):
        """Test home confinement with custom configuration."""
        config = HomeConfinementConfig(
            home_radius_meters=200.0,
            night_start_hour=23,
            night_end_hour=5,
            min_night_points=5,
            min_gps_points=10
        )
        features = AvoidanceFeatures(home_config=config)
        
        gps_data = self.create_sample_gps_data(days=3, points_per_day=24)
        result = features.home_confinement(gps_data)
        
        # Check that custom parameters were used
        params = result['processing_parameters']
        assert params['home_radius_meters'] == 200.0
        assert params['night_start_hour'] == 23
        assert params['night_end_hour'] == 5
    
    def test_home_location_detection(self):
        """Test home location detection algorithm."""
        features = AvoidanceFeatures()
        
        # Create data with clear nighttime cluster
        home_lat, home_lon = 40.7128, -74.0060
        gps_data = {
            'timestamp': [],
            'latitude': [],
            'longitude': []
        }
        
        base_time = datetime(2026, 2, 21, 0, 0, 0)
        
        # Add nighttime points around home
        for day in range(4):  # Increased to 4 days
            for hour in [23, 0, 1, 2, 3, 4, 5]:  # Nighttime hours
                for minute in range(0, 60, 30):
                    timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
                    # Add small random variation around home
                    lat = home_lat + (hash(str(timestamp)) % 20 - 10) * 0.0001
                    lon = home_lon + (hash(str(timestamp) + '1') % 20 - 10) * 0.0001
                    
                    gps_data['timestamp'].append(timestamp)
                    gps_data['latitude'].append(lat)
                    gps_data['longitude'].append(lon)
        
        # Add some daytime points away from home
        for day in range(4):  # Increased to 4 days
            for hour in range(6, 22):
                timestamp = base_time + timedelta(days=day, hours=hour)
                lat = home_lat + 0.01  # 1km away
                lon = home_lon + 0.01
                
                gps_data['timestamp'].append(timestamp)
                gps_data['latitude'].append(lat)
                gps_data['longitude'].append(lon)
        
        result = features.home_confinement(gps_data)
        
        # Should detect home
        assert result['home_location']['detected'] == True
        assert abs(result['home_location']['latitude'] - home_lat) < 0.001
        assert abs(result['home_location']['longitude'] - home_lon) < 0.001
    
    def test_gps_data_validation(self):
        """Test GPS data validation."""
        features = AvoidanceFeatures()
        
        # Missing required column
        invalid_data = {
            'timestamp': [datetime.now()],
            'latitude': [40.7128]
            # Missing 'longitude'
        }
        
        with pytest.raises(ValueError, match="Missing required column: longitude"):
            features.home_confinement(invalid_data)
        
        # Unequal column lengths
        invalid_data = {
            'timestamp': [datetime.now(), datetime.now()],
            'latitude': [40.7128],
            'longitude': [-74.0060, -73.9860]
        }
        
        with pytest.raises(ValueError, match="All GPS data columns must have equal length"):
            features.home_confinement(invalid_data)
        
        # Invalid coordinates
        invalid_coords = {
            'timestamp': [datetime.now()],
            'latitude': [91.0],  # Invalid latitude
            'longitude': [-74.0060]
        }
        
        with pytest.raises(ValueError, match="Invalid latitude"):
            features.home_confinement(invalid_coords)
    
    def test_haversine_distance_calculation(self):
        """Test haversine distance calculation."""
        features = AvoidanceFeatures()
        
        # Test known distance (NYC to LA approximately 3935 km)
        nyc_lat, nyc_lon = 40.7128, -74.0060
        la_lat, la_lon = 34.0522, -118.2437
        
        distance = features._haversine_distance(nyc_lat, nyc_lon, la_lat, la_lon)
        
        # Should be approximately 3935 km (within 5% tolerance)
        expected_distance = 3935000  # meters
        assert abs(distance - expected_distance) < expected_distance * 0.05
        
        # Test zero distance
        zero_distance = features._haversine_distance(nyc_lat, nyc_lon, nyc_lat, nyc_lon)
        assert zero_distance == 0.0


class TestCommunicationGaps:
    """Test communication gaps feature extraction."""
    
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
    
    def test_communication_gaps_basic(self):
        """Test basic communication gaps calculation."""
        config = CommunicationGapsConfig(
            min_communications=5,
            min_gap_duration_minutes=15.0
        )
        features = AvoidanceFeatures(comm_config=config)
        
        # Create communication data
        comm_data = self.create_sample_communication_data(days=5, communications_per_day=8)
        
        result = features.communication_gaps(comm_data)
        
        # Check result structure
        assert 'communication_gaps' in result
        assert 'weekly_max_gap_hours' in result
        assert 'daily_gaps' in result
        assert 'quality_metrics' in result
        assert 'processing_parameters' in result
        assert 'data_summary' in result
        
        # Check values
        assert result['weekly_max_gap_hours'] >= 0.0
        assert len(result['daily_gaps']) > 0
        assert result['communication_gaps']['total_outgoing'] >= 0
        
        # Check daily gaps structure
        for date, gap_data in result['daily_gaps'].items():
            assert 'max_gap_hours' in gap_data
            assert 'mean_gap_hours' in gap_data
            assert 'outgoing_count' in gap_data
            assert 'gap_count' in gap_data
    
    def test_communication_gaps_insufficient_data(self):
        """Test communication gaps with insufficient data."""
        config = CommunicationGapsConfig(min_communications=10)
        features = AvoidanceFeatures(comm_config=config)
        
        # Create sparse data
        sparse_data = {
            'timestamp': [datetime.now()],
            'direction': ['outgoing'],
            'contact': ['contact_1']
        }
        
        with pytest.raises(ValueError, match="Insufficient communication data"):
            features.communication_gaps(sparse_data)
    
    def test_communication_gaps_no_outgoing(self):
        """Test communication gaps with no outgoing communications."""
        config = CommunicationGapsConfig(
            min_communications=3,
            min_gap_duration_minutes=30.0
        )
        features = AvoidanceFeatures(comm_config=config)
        
        # Create data with only incoming communications
        incoming_only_data = {
            'timestamp': [datetime(2026, 2, 21, 9 + i, 0, 0) for i in range(5)],
            'direction': ['incoming'] * 5,
            'contact': [f'contact_{i}' for i in range(5)]
        }
        
        result = features.communication_gaps(incoming_only_data)
        
        # Should show full day gaps for all days
        assert result['weekly_max_gap_hours'] == 24.0
        assert result['communication_gaps']['total_outgoing'] == 0
        assert result['communication_gaps']['days_with_no_outgoing'] > 0
    
    def test_communication_gaps_single_outgoing(self):
        """Test communication gaps with single outgoing communication per day."""
        config = CommunicationGapsConfig(
            min_communications=3,
            min_gap_duration_minutes=30.0
        )
        features = AvoidanceFeatures(comm_config=config)
        
        # Create data with one outgoing communication per day
        single_outgoing_data = {
            'timestamp': [datetime(2026, 2, 21 + i, 12, 0, 0) for i in range(3)],
            'direction': ['outgoing'] * 3,
            'contact': ['contact_1'] * 3
        }
        
        result = features.communication_gaps(single_outgoing_data)
        
        # Should calculate gaps to day boundaries
        assert result['weekly_max_gap_hours'] >= 12.0  # At least 12 hours
        assert result['communication_gaps']['total_outgoing'] == 3
    
    def test_communication_data_validation(self):
        """Test communication data validation."""
        features = AvoidanceFeatures()
        
        # Missing required column
        invalid_data = {
            'timestamp': [datetime.now()],
            'direction': ['outgoing']
            # Missing 'contact'
        }
        
        with pytest.raises(ValueError, match="Missing required column: contact"):
            features.communication_gaps(invalid_data)
        
        # Invalid direction
        invalid_direction = {
            'timestamp': [datetime.now()],
            'direction': ['invalid'],
            'contact': ['contact_1']
        }
        
        with pytest.raises(ValueError, match="Invalid direction"):
            features.communication_gaps(invalid_direction)
    
    def test_communication_quality_assessment(self):
        """Test communication data quality assessment."""
        features = AvoidanceFeatures()
        
        # High quality data
        high_quality_data = self.create_sample_communication_data(days=3, communications_per_day=20)
        
        quality = features._assess_communication_quality(
            high_quality_data['timestamp'],
            high_quality_data['direction'],
            high_quality_data['contact']
        )
        
        assert quality['overall_quality'] > 0.5
        assert quality['communications_per_day'] > 10
        assert quality['direction_balance'] >= 0
        
        # Low quality data (sparse)
        low_quality_data = {
            'timestamp': [datetime(2026, 2, 21, 12, 0, 0) + timedelta(hours=i*12) for i in range(3)],
            'direction': ['outgoing', 'incoming', 'outgoing'],
            'contact': ['contact_1', 'contact_2', 'contact_1']
        }
        
        quality = features._assess_communication_quality(
            low_quality_data['timestamp'],
            low_quality_data['direction'],
            low_quality_data['contact']
        )
        
        assert quality['communications_per_day'] < 5
        assert quality['overall_quality'] < 0.7


class TestMovementRadius:
    """Test movement radius feature extraction."""
    
    def create_sample_gps_data_for_radius(self, days: int = 7, points_per_day: int = 24,
                                         center_lat: float = 40.7128, center_lon: float = -74.0060,
                                         max_radius_km: float = 5.0):
        """Create sample GPS data for movement radius calculation."""
        timestamps = []
        latitudes = []
        longitudes = []
        
        base_time = datetime(2026, 2, 21, 0, 0, 0)
        
        for day in range(days):
            for point in range(points_per_day):
                timestamp = base_time + timedelta(days=day, hours=point)
                
                # Create points within max_radius_km of center
                angle = (hash(str(day * 100 + point)) % 360) * math.pi / 180
                radius = (hash(str(day * 200 + point)) % 100) / 100 * max_radius_km
                
                # Convert to lat/lon offset (rough approximation)
                lat_offset = radius * math.cos(angle) / 111  # 1 degree ≈ 111 km
                lon_offset = radius * math.sin(angle) / 111
                
                lat = center_lat + lat_offset
                lon = center_lon + lon_offset
                
                timestamps.append(timestamp)
                latitudes.append(lat)
                longitudes.append(lon)
        
        return {
            'timestamp': timestamps,
            'latitude': latitudes,
            'longitude': longitudes
        }
    
    def test_movement_radius_basic(self):
        """Test basic movement radius calculation."""
        config = MovementRadiusConfig(
            min_gps_points=20,
            use_haversine=True,
            outlier_threshold_std=0.0  # No outlier removal
        )
        features = AvoidanceFeatures(radius_config=config)
        
        # Create GPS data within 5km radius
        gps_data = self.create_sample_gps_data_for_radius(days=5, points_per_day=20, max_radius_km=5.0)
        
        result = features.movement_radius(gps_data)
        
        # Check result structure
        assert 'movement_radius' in result
        assert 'weekly_radius_meters' in result
        assert 'center_of_mass' in result
        assert 'quality_metrics' in result
        assert 'processing_parameters' in result
        assert 'data_summary' in result
        
        # Check values
        assert result['weekly_radius_meters'] >= 0.0
        assert result['center_of_mass']['detected'] == True
        assert 'latitude' in result['center_of_mass']
        assert 'longitude' in result['center_of_mass']
        
        # Check movement radius metrics
        radius = result['movement_radius']
        assert 'radius_meters' in radius
        assert 'max_distance_meters' in radius
        assert 'mean_distance_meters' in radius
        assert radius['center_detected'] == True
    
    def test_movement_radius_insufficient_data(self):
        """Test movement radius with insufficient data."""
        config = MovementRadiusConfig(min_gps_points=50)
        features = AvoidanceFeatures(radius_config=config)
        
        # Create sparse data
        sparse_data = {
            'timestamp': [datetime.now()],
            'latitude': [40.7128],
            'longitude': [-74.0060]
        }
        
        with pytest.raises(ValueError, match="Insufficient GPS points"):
            features.movement_radius(sparse_data)
    
    def test_movement_radius_single_location(self):
        """Test movement radius with single location."""
        config = MovementRadiusConfig(
            min_gps_points=5,
            use_haversine=True
        )
        features = AvoidanceFeatures(radius_config=config)
        
        # Create data with single location
        single_location_data = {
            'timestamp': [datetime(2026, 2, 21, i, 0, 0) for i in range(10)],
            'latitude': [40.7128] * 10,
            'longitude': [-74.0060] * 10
        }
        
        result = features.movement_radius(single_location_data)
        
        # Should have zero radius (allow for floating point precision)
        assert result['weekly_radius_meters'] == 0.0
        assert result['movement_radius']['max_distance_meters'] < 1e-6
        assert result['movement_radius']['mean_distance_meters'] < 1e-6
    
    def test_movement_radius_outlier_removal(self):
        """Test movement radius with outlier removal."""
        config = MovementRadiusConfig(
            min_gps_points=10,
            use_haversine=True,
            outlier_threshold_std=2.0
        )
        features = AvoidanceFeatures(radius_config=config)
        
        # Create data with outliers
        gps_data = self.create_sample_gps_data_for_radius(days=3, points_per_day=20, max_radius_km=1.0)
        
        # Add some far outliers
        outlier_timestamps = [
            datetime(2026, 2, 21, 12, 0, 0),
            datetime(2026, 2, 22, 15, 0, 0),
            datetime(2026, 2, 23, 18, 0, 0)
        ]
        
        for ts in outlier_timestamps:
            gps_data['timestamp'].append(ts)
            gps_data['latitude'].append(40.7128 + 0.1)  # ~11km away
            gps_data['longitude'].append(-74.0060 + 0.1)
        
        result = features.movement_radius(gps_data)
        
        # Should have removed some outliers
        assert result['data_summary']['outliers_removed'] > 0
        # Radius should still be reasonable (not dominated by outliers)
        assert result['weekly_radius_meters'] < 10000  # Less than 10km
    
    def test_movement_radius_distance_methods(self):
        """Test different distance calculation methods."""
        # Create test data
        gps_data = self.create_sample_gps_data_for_radius(days=3, points_per_day=20, max_radius_km=2.0)
        
        # Test with haversine
        config_haversine = MovementRadiusConfig(
            min_gps_points=10,
            use_haversine=True,
            outlier_threshold_std=0.0
        )
        features_haversine = AvoidanceFeatures(radius_config=config_haversine)
        result_haversine = features_haversine.movement_radius(gps_data)
        
        # Test with euclidean
        config_euclidean = MovementRadiusConfig(
            min_gps_points=10,
            use_haversine=False,
            outlier_threshold_std=0.0
        )
        features_euclidean = AvoidanceFeatures(radius_config=config_euclidean)
        result_euclidean = features_euclidean.movement_radius(gps_data)
        
        # Both should produce valid results
        assert result_haversine['weekly_radius_meters'] >= 0
        assert result_euclidean['weekly_radius_meters'] >= 0
        
        # Results should be similar for small distances
        ratio = result_haversine['weekly_radius_meters'] / result_euclidean['weekly_radius_meters']
        assert 0.5 < ratio < 2.0  # Within factor of 2
    
    def test_center_of_mass_calculation(self):
        """Test center of mass calculation."""
        features = AvoidanceFeatures()
        
        # Test with known coordinates
        latitudes = [40.0, 41.0, 42.0]
        longitudes = [-74.0, -73.0, -72.0]
        
        center = features._calculate_center_of_mass(latitudes, longitudes)
        
        assert center['detected'] == True
        assert center['latitude'] == 41.0  # Mean of [40, 41, 42]
        assert center['longitude'] == -73.0  # Mean of [-74, -73, -72]
        assert center['point_count'] == 3
        
        # Test with empty data
        empty_center = features._calculate_center_of_mass([], [])
        assert empty_center['detected'] == False
    
    def test_movement_radius_calculation(self):
        """Test movement radius (radius of gyration) calculation."""
        features = AvoidanceFeatures()
        
        # Create test data with known distances
        center_lat, center_lon = 40.0, -74.0
        latitudes = [40.0, 40.01, 39.99]  # Points at different distances
        longitudes = [-74.0, -74.0, -74.0]
        
        center_of_mass = {
            'detected': True,
            'latitude': center_lat,
            'longitude': center_lon
        }
        
        # Test with haversine
        features.radius_config.use_haversine = True
        radius_haversine = features._calculate_movement_radius(latitudes, longitudes, center_of_mass)
        
        # Test with euclidean
        features.radius_config.use_haversine = False
        radius_euclidean = features._calculate_movement_radius(latitudes, longitudes, center_of_mass)
        
        # Both should produce valid results
        assert radius_haversine['radius_meters'] > 0
        assert radius_euclidean['radius_meters'] > 0
        assert radius_haversine['center_detected'] == True
        assert radius_euclidean['center_detected'] == True
        
        # Should have max distance > mean distance > radius of gyration
        assert radius_haversine['max_distance_meters'] >= radius_haversine['mean_distance_meters']
        assert radius_haversine['mean_distance_meters'] >= radius_haversine['radius_meters']
