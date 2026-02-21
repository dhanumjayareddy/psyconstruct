"""
Example usage of Social Engagement (SE) construct features.

This example demonstrates how to:
1. Extract communication frequency from communication logs
2. Calculate contact diversity from communication patterns
3. Compute initiation rates from directional communication data
4. Interpret social engagement patterns and behaviors
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psyconstruct.features.social_engagement import (
    SocialEngagementFeatures,
    CommunicationFrequencyConfig,
    ContactDiversityConfig,
    InitiationRateConfig
)


def create_socially_active_comm_data(days: int = 7):
    """Create communication data for socially active individual."""
    
    print(f"Generating {days} days of socially active communication data...")
    
    timestamps = []
    directions = []
    contacts = []
    
    # Contact pool for active individual
    contacts_pool = [f'friend_{i}' for i in range(8)] + [f'family_{i}' for i in range(4)] + [f'colleague_{i}' for i in range(5)]
    
    base_time = datetime(2026, 2, 21, 8, 0, 0)
    
    for day in range(days):
        # High communication frequency
        daily_communications = random.randint(15, 25)
        
        for comm in range(daily_communications):
            hour = 8 + random.randint(0, 14)  # 8 AM - 10 PM
            minute = random.randint(0, 59)
            timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
            
            # Balanced incoming/outgoing (slightly more outgoing for active engagement)
            direction = 'outgoing' if random.random() < 0.55 else 'incoming'
            contact = random.choice(contacts_pool)
            
            timestamps.append(timestamp)
            directions.append(direction)
            contacts.append(contact)
    
    return {
        'timestamp': timestamps,
        'direction': directions,
        'contact': contacts
    }


def create_socially_withdrawn_comm_data(days: int = 7):
    """Create communication data for socially withdrawn individual."""
    
    print(f"Generating {days} days of socially withdrawn communication data...")
    
    timestamps = []
    directions = []
    contacts = []
    
    # Limited contact pool
    contacts_pool = ['family_member_1', 'family_member_2', 'close_friend']
    
    base_time = datetime(2026, 2, 21, 8, 0, 0)
    
    for day in range(days):
        # Low communication frequency
        if random.random() < 0.6:  # 60% chance of any communication
            daily_communications = random.randint(1, 4)
            
            for comm in range(daily_communications):
                hour = 10 + random.randint(0, 8)  # 10 AM - 6 PM
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                # Mostly incoming (passive communication)
                direction = 'incoming' if random.random() < 0.8 else 'outgoing'
                contact = random.choice(contacts_pool)
                
                timestamps.append(timestamp)
                directions.append(direction)
                contacts.append(contact)
    
    return {
        'timestamp': timestamps,
        'direction': directions,
        'contact': contacts
    }


def create_professional_comm_data(days: int = 7):
    """Create communication data for professionally-focused individual."""
    
    print(f"Generating {days} days of professionally-focused communication data...")
    
    timestamps = []
    directions = []
    contacts = []
    
    # Professional contact pool
    contacts_pool = [f'colleague_{i}' for i in range(10)] + [f'client_{i}' for i in range(5)] + ['boss']
    
    base_time = datetime(2026, 2, 21, 8, 0, 0)
    
    for day in range(days):
        # Weekday vs weekend pattern
        if day < 5:  # Weekday
            daily_communications = random.randint(20, 30)
            start_hour = 8
            end_hour = 18
        else:  # Weekend
            daily_communications = random.randint(2, 6)
            start_hour = 10
            end_hour = 16
        
        for comm in range(daily_communications):
            hour = start_hour + random.randint(0, end_hour - start_hour - 1)
            minute = random.randint(0, 59)
            timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
            
            # Professional context: balanced but slightly more incoming
            direction = 'incoming' if random.random() < 0.52 else 'outgoing'
            contact = random.choice(contacts_pool)
            
            timestamps.append(timestamp)
            directions.append(direction)
            contacts.append(contact)
    
    return {
        'timestamp': timestamps,
        'direction': directions,
        'contact': contacts
    }


def example_communication_frequency_analysis():
    """Example showing communication frequency analysis."""
    print("=== Communication Frequency Analysis Example ===")
    
    # Initialize with custom configuration
    config = CommunicationFrequencyConfig(
        analysis_window_days=7,
        min_communications_per_day=1,
        include_incoming=False,  # Focus on outgoing for engagement
        smooth_daily_counts=True,
        exclude_weekends=False
    )
    features = SocialEngagementFeatures(freq_config=config)
    
    # Analyze different profiles
    profiles = [
        ("Socially Active", create_socially_active_comm_data(days=7)),
        ("Socially Withdrawn", create_socially_withdrawn_comm_data(days=7)),
        ("Professional", create_professional_comm_data(days=7))
    ]
    
    for profile_name, comm_data in profiles:
        print(f"\n{profile_name} Profile:")
        result = features.communication_frequency(comm_data)
        
        freq = result['communication_frequency']
        print(f"  Weekly outgoing: {result['weekly_outgoing_count']}")
        print(f"  Mean daily: {freq['mean_daily_frequency']:.1f}")
        print(f"  Active days: {freq['active_days']}")
        print(f"  Total days: {len(freq['daily_frequency'])}")
        
        # Show daily pattern
        print(f"  Daily pattern (first 5 days):")
        for i, (date, data) in enumerate(list(freq['daily_frequency'].items())[:5]):
            print(f"    {date}: {data['outgoing_communications']} outgoing, {data['incoming_communications']} incoming")
    
    # Comparison
    print(f"\nFrequency Comparison:")
    active_result = features.communication_frequency(create_socially_active_comm_data(days=7))
    withdrawn_result = features.communication_frequency(create_socially_withdrawn_comm_data(days=7))
    
    freq_diff = active_result['weekly_outgoing_count'] - withdrawn_result['weekly_outgoing_count']
    print(f"  Active vs Withdrawn difference: {freq_diff} communications")
    
    if freq_diff > 50:
        print(f"  Interpretation: Significant difference in social engagement")
    elif freq_diff > 20:
        print(f"  Interpretation: Moderate difference in communication patterns")
    else:
        print(f"  Interpretation: Similar communication frequency")
    
    print()


def example_contact_diversity_analysis():
    """Example showing contact diversity analysis."""
    print("=== Contact Diversity Analysis Example ===")
    
    # Initialize with custom configuration
    config = ContactDiversityConfig(
        rolling_window_days=7,
        min_interactions_per_contact=1,
        exclude_auto_messages=True,
        min_communications_total=5
    )
    features = SocialEngagementFeatures(diversity_config=config)
    
    # Analyze different profiles
    profiles = [
        ("Socially Active", create_socially_active_comm_data(days=14)),
        ("Socially Withdrawn", create_socially_withdrawn_comm_data(days=14)),
        ("Professional", create_professional_comm_data(days=14))
    ]
    
    for profile_name, comm_data in profiles:
        print(f"\n{profile_name} Profile:")
        result = features.contact_diversity(comm_data)
        
        diversity = result['contact_diversity']
        print(f"  Weekly diversity: {result['weekly_diversity']} unique contacts")
        print(f"  Mean rolling diversity: {diversity['mean_rolling_diversity']:.1f}")
        print(f"  Diversity stability: {diversity['diversity_stability']:.3f}")
        print(f"  Rolling windows: {len(result['rolling_diversity'])}")
        
        # Show top contacts
        print(f"  Top 5 contacts by interactions:")
        sorted_contacts = sorted(diversity['contact_statistics'].items(), 
                               key=lambda x: x[1]['total_communications'], reverse=True)
        
        for i, (contact, stats) in enumerate(sorted_contacts[:5]):
            print(f"    {i+1}. {contact}: {stats['total_communications']} interactions")
    
    # Comparison
    print(f"\nDiversity Comparison:")
    active_result = features.contact_diversity(create_socially_active_comm_data(days=14))
    withdrawn_result = features.contact_diversity(create_socially_withdrawn_comm_data(days=14))
    
    diversity_diff = active_result['weekly_diversity'] - withdrawn_result['weekly_diversity']
    print(f"  Active vs Withdrawn difference: {diversity_diff} unique contacts")
    
    if diversity_diff > 10:
        print(f"  Interpretation: Significant difference in social network breadth")
    elif diversity_diff > 5:
        print(f"  Interpretation: Moderate difference in contact diversity")
    else:
        print(f"  Interpretation: Similar contact diversity patterns")
    
    print()


def example_initiation_rate_analysis():
    """Example showing initiation rate analysis."""
    print("=== Initiation Rate Analysis Example ===")
    
    # Initialize with custom configuration
    config = InitiationRateConfig(
        analysis_window_days=7,
        min_total_communications=5,
        handle_zero_division="return_zero",
        include_only_bidirectional=False,
        exclude_sparse_contacts=True
    )
    features = SocialEngagementFeatures(initiation_config=config)
    
    # Analyze different profiles
    profiles = [
        ("Socially Active", create_socially_active_comm_data(days=7)),
        ("Socially Withdrawn", create_socially_withdrawn_comm_data(days=7)),
        ("Professional", create_professional_comm_data(days=7))
    ]
    
    for profile_name, comm_data in profiles:
        print(f"\n{profile_name} Profile:")
        result = features.initiation_rate(comm_data)
        
        initiation = result['initiation_rate']
        print(f"  Weekly initiation rate: {result['weekly_initiation_rate']:.3f}")
        print(f"  Direction balance: {initiation['direction_balance']:.3f}")
        print(f"  Proactive contacts: {initiation['proactive_contacts']}")
        print(f"  Total contacts analyzed: {initiation['total_contacts_analyzed']}")
        
        # Show direction counts
        counts = initiation['direction_counts']
        print(f"  Direction breakdown: {counts['outgoing']} outgoing, {counts['incoming']} incoming")
        
        # Show top proactive contacts
        print(f"  Most proactive contacts:")
        proactive_contacts = [(c, d) for c, d in result['contact_initiation_rates'].items() 
                             if d['initiation_rate'] > 0.5]
        proactive_contacts.sort(key=lambda x: x[1]['initiation_rate'], reverse=True)
        
        for i, (contact, rate_data) in enumerate(proactive_contacts[:3]):
            print(f"    {i+1}. {contact}: {rate_data['initiation_rate']:.2f} rate "
                  f"({rate_data['outgoing_count']}/{rate_data['total_count']})")
    
    # Comparison
    print(f"\nInitiation Rate Comparison:")
    active_result = features.initiation_rate(create_socially_active_comm_data(days=7))
    withdrawn_result = features.initiation_rate(create_socially_withdrawn_comm_data(days=7))
    
    rate_diff = active_result['weekly_initiation_rate'] - withdrawn_result['weekly_initiation_rate']
    print(f"  Active vs Withdrawn difference: {rate_diff:.3f}")
    
    if rate_diff > 0.2:
        print(f"  Interpretation: Significant difference in proactive engagement")
    elif rate_diff > 0.1:
        print(f"  Interpretation: Moderate difference in initiation patterns")
    else:
        print(f"  Interpretation: Similar initiation rates")
    
    print()


def example_social_engagement_profile():
    """Example showing complete social engagement profile analysis."""
    print("=== Complete Social Engagement Profile Analysis ===")
    
    # Initialize all features with research-grade configuration
    freq_config = CommunicationFrequencyConfig(
        analysis_window_days=14,
        min_communications_per_day=1,
        include_incoming=False,
        smooth_daily_counts=True
    )
    
    diversity_config = ContactDiversityConfig(
        rolling_window_days=7,
        min_interactions_per_contact=2,
        exclude_auto_messages=True,
        min_communications_total=10
    )
    
    initiation_config = InitiationRateConfig(
        analysis_window_days=14,
        min_total_communications=10,
        handle_zero_division="return_zero",
        exclude_sparse_contacts=True
    )
    
    features = SocialEngagementFeatures(
        freq_config=freq_config,
        diversity_config=diversity_config,
        initiation_config=initiation_config
    )
    
    # Analyze different profiles
    profiles = [
        ("Highly Social", create_socially_active_comm_data(days=14)),
        ("Socially Withdrawn", create_socially_withdrawn_comm_data(days=14)),
        ("Professional", create_professional_comm_data(days=14))
    ]
    
    for profile_name, comm_data in profiles:
        print(f"\n{profile_name} Profile:")
        print("-" * 50)
        
        # Extract all features
        freq_result = features.communication_frequency(comm_data)
        diversity_result = features.contact_diversity(comm_data)
        initiation_result = features.initiation_rate(comm_data)
        
        # Calculate engagement scores (0-100, higher = more engaged)
        frequency_score = min(freq_result['weekly_outgoing_count'] / 50 * 100, 100)  # 50/week as reference
        diversity_score = min(diversity_result['weekly_diversity'] / 20 * 100, 100)  # 20 contacts as reference
        initiation_score = initiation_result['weekly_initiation_rate'] * 100
        
        overall_engagement = (frequency_score + diversity_score + initiation_score) / 3
        
        print(f"Communication Frequency: {frequency_score:.1f}%")
        print(f"Contact Diversity: {diversity_score:.1f}%")
        print(f"Initiation Rate: {initiation_score:.1f}%")
        print(f"Overall Engagement: {overall_engagement:.1f}%")
        
        # Behavioral interpretation
        if overall_engagement > 70:
            behavior = "High social engagement - very active and connected"
        elif overall_engagement > 40:
            behavior = "Moderate social engagement - typical social patterns"
        else:
            behavior = "Low social engagement - socially withdrawn or isolated"
        
        print(f"Behavioral Pattern: {behavior}")
        
        # Engagement balance analysis
        if initiation_score > 60 and diversity_score > 60:
            balance = "Proactive and diverse engagement"
        elif initiation_score > 60:
            balance = "Proactive but focused engagement"
        elif diversity_score > 60:
            balance = "Reactive but diverse engagement"
        else:
            balance = "Limited or passive engagement"
        
        print(f"Engagement Balance: {balance}")
        
        # Quality indicators
        freq_quality = freq_result['quality_metrics']['overall_quality']
        diversity_quality = diversity_result['quality_metrics']['overall_quality']
        initiation_quality = initiation_result['quality_metrics']['overall_quality']
        
        print(f"Data Quality: Freq={freq_quality:.2f}, Div={diversity_quality:.2f}, Init={initiation_quality:.2f}")
    
    print()


def example_clinical_interpretation():
    """Example showing clinical interpretation of social engagement features."""
    print("=== Clinical Interpretation Example ===")
    
    features = SocialEngagementFeatures()
    
    # Create clinical scenarios
    scenarios = [
        {
            'name': 'Depressive Withdrawal',
            'comm_data': create_socially_withdrawn_comm_data(days=14),
            'description': 'Patient shows reduced communication frequency and diversity'
        },
        {
            'name': 'Social Anxiety',
            'comm_data': create_professional_comm_data(days=14),  # Professional but limited social
            'description': 'Patient maintains professional contacts but limited social engagement'
        },
        {
            'name': 'Healthy Baseline',
            'comm_data': create_socially_active_comm_data(days=14),
            'description': 'Normal social engagement patterns across multiple domains'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"Description: {scenario['description']}")
        print("-" * 60)
        
        # Extract features
        freq_result = features.communication_frequency(scenario['comm_data'])
        diversity_result = features.contact_diversity(scenario['comm_data'])
        initiation_result = features.initiation_rate(scenario['comm_data'])
        
        # Clinical metrics
        weekly_outgoing = freq_result['weekly_outgoing_count']
        diversity = diversity_result['weekly_diversity']
        initiation_rate = initiation_result['weekly_initiation_rate']
        
        print(f"Weekly Outgoing: {weekly_outgoing}")
        print(f"Contact Diversity: {diversity}")
        print(f"Initiation Rate: {initiation_rate:.3f}")
        
        # Clinical interpretation
        print(f"\nClinical Assessment:")
        
        if weekly_outgoing < 10:
            print(f"  ✓ Low communication frequency suggests social withdrawal")
        elif weekly_outgoing < 25:
            print(f"  ⚠ Moderate communication frequency may indicate emerging withdrawal")
        else:
            print(f"  ✓ Normal communication frequency")
        
        if diversity < 5:
            print(f"  ✓ Low contact diversity indicates limited social network")
        elif diversity < 10:
            print(f"  ⚠ Moderate diversity may suggest social constriction")
        else:
            print(f"  ✓ Normal contact diversity")
        
        if initiation_rate < 0.3:
            print(f"  ✓ Low initiation rate indicates passive social behavior")
        elif initiation_rate < 0.5:
            print(f"  ⚠ Moderate initiation rate may reflect reduced proactivity")
        else:
            print(f"  ✓ Normal initiation patterns")
        
        # Risk assessment
        risk_factors = 0
        if weekly_outgoing < 10:
            risk_factors += 1
        if diversity < 5:
            risk_factors += 1
        if initiation_rate < 0.3:
            risk_factors += 1
        
        if risk_factors >= 2:
            risk_level = "HIGH - Significant social impairment"
        elif risk_factors == 1:
            risk_level = "MODERATE - Monitor social functioning"
        else:
            risk_level = "LOW - Healthy social engagement"
        
        print(f"\nRisk Level: {risk_level}")
        
        # Treatment implications
        if risk_factors >= 2:
            print(f"Treatment Focus: Social skills training, behavioral activation")
        elif risk_factors == 1:
            print(f"Treatment Focus: Monitor and encourage social activities")
        else:
            print(f"Treatment Focus: Maintain healthy social patterns")
    
    print()


def example_research_configuration():
    """Example showing research-grade configuration."""
    print("=== Research Configuration Examples ===")
    
    # Research configuration (high precision)
    research_config = {
        'freq_config': CommunicationFrequencyConfig(
            analysis_window_days=14,      # 2-week analysis
            min_communications_per_day=1,  # Sensitive threshold
            include_incoming=True,         # Include all communications
            smooth_daily_counts=False,     # No smoothing for raw data
            exclude_weekends=False         # Include all days
        ),
        'diversity_config': ContactDiversityConfig(
            rolling_window_days=7,         # Standard rolling window
            min_interactions_per_contact=2, # Filter brief contacts
            exclude_auto_messages=True,    # Remove automated messages
            min_communications_total=20     # Higher threshold
        ),
        'initiation_config': InitiationRateConfig(
            analysis_window_days=14,      # 2-week analysis
            min_total_communications=20,   # Higher threshold
            handle_zero_division="return_nan",  # Statistical handling
            include_only_bidirectional=True, # Focus on reciprocal contacts
            exclude_sparse_contacts=True   # Remove noise
        )
    }
    
    print("Research Configuration (High Precision):")
    print(f"  Communication Frequency:")
    print(f"    Analysis window: {research_config['freq_config'].analysis_window_days} days")
    print(f"    Min communications/day: {research_config['freq_config'].min_communications_per_day}")
    print(f"    Include incoming: {research_config['freq_config'].include_incoming}")
    print(f"    Smoothing: {research_config['freq_config'].smooth_daily_counts}")
    
    print(f"  Contact Diversity:")
    print(f"    Rolling window: {research_config['diversity_config'].rolling_window_days} days")
    print(f"    Min interactions/contact: {research_config['diversity_config'].min_interactions_per_contact}")
    print(f"    Exclude auto messages: {research_config['diversity_config'].exclude_auto_messages}")
    
    print(f"  Initiation Rate:")
    print(f"    Zero division handling: {research_config['initiation_config'].handle_zero_division}")
    print(f"    Bidirectional only: {research_config['initiation_config'].include_only_bidirectional}")
    print(f"    Exclude sparse contacts: {research_config['initiation_config'].exclude_sparse_contacts}")
    
    # Test research configuration
    print(f"\nTesting Research Configuration:")
    research_features = SocialEngagementFeatures(**research_config)
    
    # Use high-quality data
    research_data = create_socially_active_comm_data(days=14)
    
    try:
        freq_result = research_features.communication_frequency(research_data)
        print(f"  ✓ Communication frequency: {freq_result['weekly_outgoing_count']} outgoing")
    except ValueError as e:
        print(f"  ✗ Communication frequency failed: {e}")
    
    try:
        diversity_result = research_features.contact_diversity(research_data)
        print(f"  ✓ Contact diversity: {diversity_result['weekly_diversity']} contacts")
    except ValueError as e:
        print(f"  ✗ Contact diversity failed: {e}")
    
    try:
        initiation_result = research_features.initiation_rate(research_data)
        print(f"  ✓ Initiation rate: {initiation_result['weekly_initiation_rate']:.3f}")
    except ValueError as e:
        print(f"  ✗ Initiation rate failed: {e}")
    
    print()


if __name__ == "__main__":
    """Run all social engagement examples."""
    print("Psyconstruct Social Engagement (SE) Construct Examples")
    print("=" * 60)
    
    example_communication_frequency_analysis()
    example_contact_diversity_analysis()
    example_initiation_rate_analysis()
    example_social_engagement_profile()
    example_clinical_interpretation()
    example_research_configuration()
    
    print("All social engagement examples completed successfully!")
