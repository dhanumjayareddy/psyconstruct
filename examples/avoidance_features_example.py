"""
Example usage of Avoidance (AV) construct features.

This example demonstrates how to:
1. Extract home confinement from GPS data
2. Calculate communication gaps from communication logs
3. Compute movement radius from GPS coordinates
4. Interpret avoidance patterns and behaviors
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psyconstruct.features.avoidance import (
    AvoidanceFeatures,
    HomeConfinementConfig,
    CommunicationGapsConfig,
    MovementRadiusConfig
)


def create_home_bound_gps_data(days: int = 7, home_lat: float = 40.7128, home_lon: float = -74.0060):
    """Create GPS data for someone who stays mostly at home."""
    
    print(f"Generating {days} days of home-bound GPS data...")
    
    timestamps = []
    latitudes = []
    longitudes = []
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    for day in range(days):
        for hour in range(24):
            for minute in range(0, 60, 30):  # Every 30 minutes
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
                
                # 90% chance of being at home, 10% chance of short outing
                if random.random() < 0.9:
                    # At home - small random variation around home coordinates
                    lat = home_lat + (random.random() - 0.5) * 0.0001
                    lon = home_lon + (random.random() - 0.5) * 0.0001
                else:
                    # Short outing - within 1km of home
                    angle = random.random() * 2 * math.pi
                    distance = random.random() * 1000  # 0-1km
                    
                    lat_offset = distance * math.cos(angle) / 111000
                    lon_offset = distance * math.sin(angle) / 111000
                    
                    lat = home_lat + lat_offset
                    lon = home_lon + lon_offset
                
                timestamps.append(timestamp)
                latitudes.append(lat)
                longitudes.append(lon)
    
    return {
        'timestamp': timestamps,
        'latitude': latitudes,
        'longitude': longitudes
    }


def create_social_gps_data(days: int = 7, home_lat: float = 40.7128, home_lon: float = -74.0060):
    """Create GPS data for someone with regular social activities."""
    
    print(f"Generating {days} days of socially active GPS data...")
    
    timestamps = []
    latitudes = []
    longitudes = []
    
    # Define activity locations
    locations = {
        'home': (home_lat, home_lon),
        'work': (home_lat + 0.01, home_lon + 0.01),  # ~1.4km away
        'gym': (home_lat - 0.008, home_lon + 0.012),   # ~1.5km away
        'friends': (home_lat + 0.015, home_lon - 0.005), # ~1.7km away
        'shopping': (home_lat - 0.005, home_lon - 0.015) # ~1.7km away
    }
    
    base_time = datetime(2026, 2, 21, 0, 0, 0)
    
    for day in range(days):
        for hour in range(24):
            for minute in range(0, 60, 30):  # Every 30 minutes
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
                
                # Determine location based on time and day
                if hour >= 22 or hour < 6:  # Night - home
                    location = 'home'
                elif 6 <= hour < 9:  # Morning - home or gym
                    location = 'gym' if random.random() < 0.3 else 'home'
                elif 9 <= hour < 17:  # Workday - work
                    location = 'work' if day < 5 else 'home'  # Weekends at home
                elif 17 <= hour < 19:  # Evening - home or friends
                    location = 'friends' if random.random() < 0.4 else 'home'
                elif 19 <= hour < 21:  # Late evening - home or shopping
                    location = 'shopping' if random.random() < 0.3 else 'home'
                else:  # 21-22 - home
                    location = 'home'
                
                lat, lon = locations[location]
                
                # Add small random variation
                lat += (random.random() - 0.5) * 0.0002
                lon += (random.random() - 0.5) * 0.0002
                
                timestamps.append(timestamp)
                latitudes.append(lat)
                longitudes.append(lon)
    
    return {
        'timestamp': timestamps,
        'latitude': latitudes,
        'longitude': longitudes
    }


def create_isolated_communication_data(days: int = 7):
    """Create communication data for socially isolated individual."""
    
    print(f"Generating {days} days of isolated communication data...")
    
    timestamps = []
    directions = []
    contacts = []
    
    base_time = datetime(2026, 2, 21, 8, 0, 0)
    
    for day in range(days):
        # Very few communications, mostly incoming
        if random.random() < 0.3:  # 30% chance of any communication
            # 1-2 communications per day
            num_communications = random.randint(1, 2)
            
            for i in range(num_communications):
                hour = 10 + random.randint(0, 8)  # 10 AM - 6 PM
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                # 80% incoming, 20% outgoing
                direction = 'incoming' if random.random() < 0.8 else 'outgoing'
                contact = 'family_member' if direction == 'incoming' else 'emergency_contact'
                
                timestamps.append(timestamp)
                directions.append(direction)
                contacts.append(contact)
    
    return {
        'timestamp': timestamps,
        'direction': directions,
        'contact': contacts
    }


def create_social_communication_data(days: int = 7):
    """Create communication data for socially active individual."""
    
    print(f"Generating {days} days of socially active communication data...")
    
    timestamps = []
    directions = []
    contacts = []
    
    # Contact pool
    contacts_pool = [f'friend_{i}' for i in range(5)] + [f'family_{i}' for i in range(3)] + ['colleague_1', 'colleague_2']
    
    base_time = datetime(2026, 2, 21, 8, 0, 0)
    
    for day in range(days):
        # Regular communication pattern
        for hour in range(8, 22):  # 8 AM - 10 PM
            if random.random() < 0.4:  # 40% chance of communication each hour
                minute = random.randint(0, 59)
                timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
                
                # Balanced incoming/outgoing
                direction = 'outgoing' if random.random() < 0.5 else 'incoming'
                contact = random.choice(contacts_pool)
                
                timestamps.append(timestamp)
                directions.append(direction)
                contacts.append(contact)
    
    return {
        'timestamp': timestamps,
        'direction': directions,
        'contact': contacts
    }


def example_home_confinement_analysis():
    """Example showing home confinement analysis."""
    print("=== Home Confinement Analysis Example ===")
    
    # Initialize with default configuration
    features = AvoidanceFeatures()
    
    # Create contrasting scenarios
    print("\nScenario 1: Home-bound individual")
    home_bound_data = create_home_bound_gps_data(days=7)
    
    home_result = features.home_confinement(home_bound_data)
    
    print(f"  Home confinement: {home_result['weekly_confinement_percentage']:.1f}%")
    print(f"  Home detected: {home_result['home_location']['detected']}")
    print(f"  Points within home: {home_result['home_confinement']['points_within_home']}")
    print(f"  Total points: {home_result['home_confinement']['total_points']}")
    
    if home_result['home_location']['detected']:
        print(f"  Home location: {home_result['home_location']['latitude']:.4f}, {home_result['home_location']['longitude']:.4f}")
        print(f"  Detection confidence: {home_result['home_location']['confidence']:.2f}")
    
    print("\nScenario 2: Socially active individual")
    social_data = create_social_gps_data(days=7)
    
    social_result = features.home_confinement(social_data)
    
    print(f"  Home confinement: {social_result['weekly_confinement_percentage']:.1f}%")
    print(f"  Home detected: {social_result['home_location']['detected']}")
    print(f"  Points within home: {social_result['home_confinement']['points_within_home']}")
    print(f"  Total points: {social_result['home_confinement']['total_points']}")
    
    # Comparison
    print(f"\nComparison:")
    confinement_diff = home_result['weekly_confinement_percentage'] - social_result['weekly_confinement_percentage']
    print(f"  Difference: {confinement_diff:.1f}%")
    
    if confinement_diff > 50:
        print(f"  Interpretation: Home-bound individual shows significantly higher home confinement")
    elif confinement_diff > 20:
        print(f"  Interpretation: Moderate difference in home confinement patterns")
    else:
        print(f"  Interpretation: Similar home confinement patterns")
    
    print()


def example_communication_gaps_analysis():
    """Example showing communication gaps analysis."""
    print("=== Communication Gaps Analysis Example ===")
    
    # Initialize with custom configuration
    config = CommunicationGapsConfig(
        min_communications=1,
        min_gap_duration_minutes=15.0,
        analysis_window_days=7
    )
    features = AvoidanceFeatures(comm_config=config)
    
    # Create contrasting scenarios
    print("\nScenario 1: Socially isolated individual")
    isolated_data = create_isolated_communication_data(days=7)
    
    isolated_result = features.communication_gaps(isolated_data)
    
    print(f"  Weekly max gap: {isolated_result['weekly_max_gap_hours']:.1f} hours")
    print(f"  Mean daily gap: {isolated_result['communication_gaps']['mean_daily_gap_hours']:.1f} hours")
    print(f"  Days with no outgoing: {isolated_result['communication_gaps']['days_with_no_outgoing']}")
    print(f"  Total outgoing: {isolated_result['communication_gaps']['total_outgoing']}")
    
    print("\nScenario 2: Socially active individual")
    social_data = create_social_communication_data(days=7)
    
    social_result = features.communication_gaps(social_data)
    
    print(f"  Weekly max gap: {social_result['weekly_max_gap_hours']:.1f} hours")
    print(f"  Mean daily gap: {social_result['communication_gaps']['mean_daily_gap_hours']:.1f} hours")
    print(f"  Days with no outgoing: {social_result['communication_gaps']['days_with_no_outgoing']}")
    print(f"  Total outgoing: {social_result['communication_gaps']['total_outgoing']}")
    
    # Show daily patterns
    print(f"\nDaily gap patterns (isolated vs social):")
    isolated_daily = isolated_result['daily_gaps']
    social_daily = social_result['daily_gaps']
    
    for date in sorted(isolated_daily.keys())[:3]:  # Show first 3 days
        isolated_gap = isolated_daily[date]['max_gap_hours']
        social_gap = social_daily.get(date, {'max_gap_hours': 0})['max_gap_hours']
        
        print(f"  {date}: Isolated: {isolated_gap:.1f}h, Social: {social_gap:.1f}h")
    
    # Comparison
    print(f"\nComparison:")
    gap_diff = isolated_result['weekly_max_gap_hours'] - social_result['weekly_max_gap_hours']
    print(f"  Max gap difference: {gap_diff:.1f} hours")
    
    if gap_diff > 12:
        print(f"  Interpretation: Isolated individual shows much longer communication gaps")
    elif gap_diff > 4:
        print(f"  Interpretation: Moderate difference in communication patterns")
    else:
        print(f"  Interpretation: Similar communication gap patterns")
    
    print()


def example_movement_radius_analysis():
    """Example showing movement radius analysis."""
    print("=== Movement Radius Analysis Example ===")
    
    # Initialize with custom configuration
    config = MovementRadiusConfig(
        min_gps_points=20,
        use_haversine=True,
        outlier_threshold_std=2.0,
        analysis_window_days=7
    )
    features = AvoidanceFeatures(radius_config=config)
    
    # Create contrasting scenarios
    print("\nScenario 1: Home-bound individual")
    home_bound_data = create_home_bound_gps_data(days=7)
    
    home_result = features.movement_radius(home_bound_data)
    
    print(f"  Movement radius: {home_result['weekly_radius_meters']:.1f} meters")
    print(f"  Max distance: {home_result['movement_radius']['max_distance_meters']:.1f} meters")
    print(f"  Mean distance: {home_result['movement_radius']['mean_distance_meters']:.1f} meters")
    print(f"  Center detected: {home_result['movement_radius']['center_detected']}")
    
    if home_result['center_of_mass']['detected']:
        print(f"  Center of mass: {home_result['center_of_mass']['latitude']:.4f}, {home_result['center_of_mass']['longitude']:.4f}")
    
    print("\nScenario 2: Socially active individual")
    social_data = create_social_gps_data(days=7)
    
    social_result = features.movement_radius(social_data)
    
    print(f"  Movement radius: {social_result['weekly_radius_meters']:.1f} meters")
    print(f"  Max distance: {social_result['movement_radius']['max_distance_meters']:.1f} meters")
    print(f"  Mean distance: {social_result['movement_radius']['mean_distance_meters']:.1f} meters")
    print(f"  Center detected: {social_result['movement_radius']['center_detected']}")
    
    # Comparison
    print(f"\nComparison:")
    radius_diff = social_result['weekly_radius_meters'] - home_result['weekly_radius_meters']
    print(f"  Radius difference: {radius_diff:.1f} meters")
    
    if radius_diff > 1000:
        print(f"  Interpretation: Social individual has significantly larger movement radius")
    elif radius_diff > 500:
        print(f"  Interpretation: Moderate difference in movement patterns")
    else:
        print(f"  Interpretation: Similar movement radius patterns")
    
    print()


def example_avoidance_profile_analysis():
    """Example showing complete avoidance profile analysis."""
    print("=== Complete Avoidance Profile Analysis ===")
    
    # Initialize all features
    home_config = HomeConfinementConfig(min_night_points=5, min_gps_points=20)
    comm_config = CommunicationGapsConfig(min_communications=1, min_gap_duration_minutes=15.0)
    radius_config = MovementRadiusConfig(min_gps_points=15, outlier_threshold_std=2.0)
    
    features = AvoidanceFeatures(
        home_config=home_config,
        comm_config=comm_config,
        radius_config=radius_config
    )
    
    # Analyze different profiles
    profiles = [
        ("Home-bound Isolated", create_home_bound_gps_data(days=7), create_isolated_communication_data(days=7)),
        ("Socially Active", create_social_gps_data(days=7), create_social_communication_data(days=7))
    ]
    
    for profile_name, gps_data, comm_data in profiles:
        print(f"\n{profile_name} Profile:")
        print("-" * 40)
        
        # Extract all features
        home_result = features.home_confinement(gps_data)
        comm_result = features.communication_gaps(comm_data)
        radius_result = features.movement_radius(gps_data)
        
        # Calculate avoidance score (0-100, higher = more avoidance)
        home_score = home_result['weekly_confinement_percentage']
        comm_score = min(comm_result['weekly_max_gap_hours'] / 24 * 100, 100)  # Convert hours to percentage
        radius_score = max(0, 100 - (radius_result['weekly_radius_meters'] / 50))  # Inverse relationship
        
        overall_avoidance = (home_score + comm_score + radius_score) / 3
        
        print(f"Home Confinement: {home_score:.1f}%")
        print(f"Communication Gaps: {comm_score:.1f}%")
        print(f"Limited Movement: {radius_score:.1f}%")
        print(f"Overall Avoidance: {overall_avoidance:.1f}%")
        
        # Behavioral interpretation
        if overall_avoidance > 70:
            behavior = "High avoidance - significant social withdrawal"
        elif overall_avoidance > 40:
            behavior = "Moderate avoidance - some social limitations"
        else:
            behavior = "Low avoidance - socially engaged and active"
        
        print(f"Behavioral Pattern: {behavior}")
        
        # Quality indicators
        home_quality = home_result['quality_metrics']['overall_quality']
        comm_quality = comm_result['quality_metrics']['overall_quality']
        radius_quality = radius_result['quality_metrics']['overall_quality']
        
        print(f"Data Quality: Home={home_quality:.2f}, Comm={comm_quality:.2f}, Radius={radius_quality:.2f}")
    
    print()


def example_clinical_interpretation():
    """Example showing clinical interpretation of avoidance features."""
    print("=== Clinical Interpretation Example ===")
    
    features = AvoidanceFeatures()
    
    # Create clinical scenarios
    scenarios = [
        {
            'name': 'Depressive Withdrawal',
            'gps_data': create_home_bound_gps_data(days=14),
            'comm_data': create_isolated_communication_data(days=14),
            'description': 'Patient shows increased home confinement and reduced communication'
        },
        {
            'name': 'Social Anxiety',
            'gps_data': create_home_bound_gps_data(days=14),
            'comm_data': create_social_communication_data(days=14),  # Digital communication but physical avoidance
            'description': 'Patient maintains digital contact but avoids physical locations'
        },
        {
            'name': 'Healthy Baseline',
            'gps_data': create_social_gps_data(days=14),
            'comm_data': create_social_communication_data(days=14),
            'description': 'Normal social and physical activity patterns'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"Description: {scenario['description']}")
        print("-" * 50)
        
        # Extract features
        home_result = features.home_confinement(scenario['gps_data'])
        comm_result = features.communication_gaps(scenario['comm_data'])
        radius_result = features.movement_radius(scenario['gps_data'])
        
        # Clinical metrics
        confinement = home_result['weekly_confinement_percentage']
        max_gap = comm_result['weekly_max_gap_hours']
        radius = radius_result['weekly_radius_meters']
        
        print(f"Home Confinement: {confinement:.1f}%")
        print(f"Communication Gap: {max_gap:.1f} hours")
        print(f"Movement Radius: {radius:.1f} meters")
        
        # Clinical interpretation
        print(f"\nClinical Assessment:")
        
        if confinement > 80:
            print(f"  ✓ High home confinement suggests significant withdrawal")
        elif confinement > 60:
            print(f"  ⚠ Moderate home confinement may indicate emerging avoidance")
        else:
            print(f"  ✓ Normal home activity levels")
        
        if max_gap > 16:
            print(f"  ✓ Extended communication gaps indicate social isolation")
        elif max_gap > 8:
            print(f"  ⚠ Moderate communication gaps warrant monitoring")
        else:
            print(f"  ✓ Normal communication patterns")
        
        if radius < 500:
            print(f"  ✓ Very limited movement radius suggests agoraphobic tendencies")
        elif radius < 1500:
            print(f"  ⚠ Reduced movement may indicate avoidance of unfamiliar places")
        else:
            print(f"  ✓ Normal movement patterns")
        
        # Risk assessment
        risk_factors = 0
        if confinement > 80:
            risk_factors += 1
        if max_gap > 16:
            risk_factors += 1
        if radius < 500:
            risk_factors += 1
        
        if risk_factors >= 2:
            risk_level = "HIGH - Immediate clinical attention recommended"
        elif risk_factors == 1:
            risk_level = "MODERATE - Monitor and consider intervention"
        else:
            risk_level = "LOW - Healthy engagement patterns"
        
        print(f"\nRisk Level: {risk_level}")
    
    print()


def example_custom_configuration():
    """Example showing custom configuration for different use cases."""
    print("=== Custom Configuration Examples ===")
    
    # Research configuration (high precision)
    research_config = {
        'home_config': HomeConfinementConfig(
            home_radius_meters=25.0,  # Smaller home radius
            min_night_points=20,     # More data required
            min_gps_points=200,       # Higher data requirements
            night_start_hour=23,      # Later night start
            night_end_hour=5         # Earlier night end
        ),
        'comm_config': CommunicationGapsConfig(
            min_gap_duration_minutes=15.0,  # Smaller gaps
            min_communications=5,          # More communications required
            analysis_window_days=14        # Longer analysis
        ),
        'radius_config': MovementRadiusConfig(
            outlier_threshold_std=2.0,      # Remove outliers
            use_haversine=True,            # Precise distance calculation
            min_gps_points=100             # More points required
        )
    }
    
    print("Research Configuration (High Precision):")
    print(f"  Home radius: {research_config['home_config'].home_radius_meters}m")
    print(f"  Min night points: {research_config['home_config'].min_night_points}")
    print(f"  Min gap duration: {research_config['comm_config'].min_gap_duration_minutes}min")
    print(f"  Outlier threshold: {research_config['radius_config'].outlier_threshold_std}σ")
    
    # Clinical configuration (robust)
    clinical_config = {
        'home_config': HomeConfinementConfig(
            home_radius_meters=100.0,  # Larger home radius
            min_night_points=5,        # Fewer points required
            min_gps_points=50,        # Lower data requirements
            max_gap_hours=8.0         # Allow data gaps
        ),
        'comm_config': CommunicationGapsConfig(
            min_gap_duration_minutes=60.0,  # Larger gaps only
            min_communications=2,           # Fewer communications required
            analysis_window_days=7          # Standard week
        ),
        'radius_config': MovementRadiusConfig(
            outlier_threshold_std=3.0,      # More permissive outlier handling
            use_haversine=True,            # Still use accurate calculation
            min_gps_points=30              # Moderate data requirements
        )
    }
    
    print("\nClinical Configuration (Robust):")
    print(f"  Home radius: {clinical_config['home_config'].home_radius_meters}m")
    print(f"  Min night points: {clinical_config['home_config'].min_night_points}")
    print(f"  Min gap duration: {clinical_config['comm_config'].min_gap_duration_minutes}min")
    print(f"  Outlier threshold: {clinical_config['radius_config'].outlier_threshold_std}σ")
    
    # Test with research configuration
    print("\nTesting Research Configuration:")
    research_features = AvoidanceFeatures(**research_config)
    
    # Use higher quality data
    research_gps = create_social_gps_data(days=14)
    research_comm = create_social_communication_data(days=14)
    
    try:
        home_result = research_features.home_confinement(research_gps)
        print(f"  ✓ Home confinement: {home_result['weekly_confinement_percentage']:.1f}%")
    except ValueError as e:
        print(f"  ✗ Home confinement failed: {e}")
    
    try:
        comm_result = research_features.communication_gaps(research_comm)
        print(f"  ✓ Communication gaps: {comm_result['weekly_max_gap_hours']:.1f}h")
    except ValueError as e:
        print(f"  ✗ Communication gaps failed: {e}")
    
    try:
        radius_result = research_features.movement_radius(research_gps)
        print(f"  ✓ Movement radius: {radius_result['weekly_radius_meters']:.1f}m")
    except ValueError as e:
        print(f"  ✗ Movement radius failed: {e}")
    
    print()


if __name__ == "__main__":
    """Run all avoidance features examples."""
    print("Psyconstruct Avoidance (AV) Construct Examples")
    print("=" * 60)
    
    example_home_confinement_analysis()
    example_communication_gaps_analysis()
    example_movement_radius_analysis()
    example_avoidance_profile_analysis()
    example_clinical_interpretation()
    example_custom_configuration()
    
    print("All avoidance features examples completed successfully!")
