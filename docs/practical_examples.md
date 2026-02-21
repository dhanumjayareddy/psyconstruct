# Practical Examples and Use Cases

## Table of Contents
1. [Quick Start Examples](#quick-start-examples)
2. [Clinical Use Cases](#clinical-use-cases)
3. [Research Applications](#research-applications)
4. [Advanced Configurations](#advanced-configurations)
5. [Integration Examples](#integration-examples)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Quick Start Examples

### Basic Usage

#### Simple Behavioral Activation Analysis

```python
from psyconstruct.features import BehavioralActivationFeatures
from psyconstruct.constructs import ConstructAggregator, AggregationConfig
import pandas as pd

# Sample accelerometer data
accelerometer_data = {
    'timestamp': pd.date_range('2026-02-21', periods=1440, freq='1min'),
    'x': [0.1] * 1440,  # Simplified constant values
    'y': [0.2] * 1440,
    'z': [9.8] * 1440
}

# Initialize feature extractor
ba_features = BehavioralActivationFeatures()

# Extract activity volume
activity_result = ba_features.activity_volume(accelerometer_data)
print(f"Weekly activity: {activity_result['weekly_activity_count']}")
print(f"Quality score: {activity_result['quality_metrics']['overall_quality']}")
```

#### Complete Construct Analysis

```python
from psyconstruct import analyze_participant

# Complete participant data
participant_data = {
    'accelerometer_data': accelerometer_data,
    'gps_data': gps_data,
    'communication_data': communication_data,
    'app_usage_data': app_usage_data,
    'screen_state_data': screen_state_data
}

# Analyze all constructs
results = analyze_participant(
    participant_id='participant_001',
    **participant_data
)

# Print construct scores
for construct_name, construct_score in results.items():
    print(f"{construct_name}: {construct_score.normalized_score:.2f}")
    print(f"  Interpretation: {construct_score.interpretation}")
```

### Data Preparation Examples

#### GPS Data Preparation

```python
import pandas as pd
import numpy as np

def prepare_gps_data(raw_gps_file):
    """Prepare GPS data for Psyconstruct analysis."""
    
    # Load raw GPS data
    df = pd.read_csv(raw_gps_file)
    
    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for accuracy
    df = df[df['accuracy'] <= 50]  # Keep points with ≤50m accuracy
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Convert to required format
    gps_data = {
        'timestamp': df['timestamp'].tolist(),
        'latitude': df['latitude'].tolist(),
        'longitude': df['longitude'].tolist(),
        'accuracy': df['accuracy'].tolist()
    }
    
    return gps_data

# Usage
gps_data = prepare_gps_data('raw_gps_data.csv')
```

#### Communication Data Preparation

```python
def prepare_communication_data(call_logs, sms_logs, email_logs):
    """Combine multiple communication sources."""
    
    all_communications = []
    
    # Process call logs
    for call in call_logs:
        all_communications.append({
            'timestamp': call['timestamp'],
            'direction': call['direction'],
            'contact': call['contact'],
            'type': 'call'
        })
    
    # Process SMS logs
    for sms in sms_logs:
        all_communications.append({
            'timestamp': sms['timestamp'],
            'direction': sms['direction'],
            'contact': sms['contact'],
            'type': 'text'
        })
    
    # Process email logs
    for email in email_logs:
        all_communications.append({
            'timestamp': email['timestamp'],
            'direction': email['direction'],
            'contact': email['contact'],
            'type': 'email'
        })
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(all_communications)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Convert to required format
    communication_data = {
        'timestamp': df['timestamp'].tolist(),
        'direction': df['direction'].tolist(),
        'contact': df['contact'].tolist(),
        'type': df['type'].tolist()
    }
    
    return communication_data
```

---

## Clinical Use Cases

### Depression Screening

#### Primary Care Setting

```python
from psyconstruct import analyze_participant
from psyconstruct.utils import clinical_interpretation

def depression_screening(patient_data):
    """Screen for depression in primary care setting."""
    
    # Analyze digital phenotyping data
    construct_scores = analyze_participant(**patient_data, participant_id=patient_data['patient_id'])
    
    # Extract relevant constructs
    ba_score = construct_scores['behavioral_activation'].normalized_score
    se_score = construct_scores['social_engagement'].normalized_score
    rs_score = construct_scores['routine_stability'].normalized_score
    
    # Calculate depression risk
    depression_risk = calculate_depression_risk(ba_score, se_score, rs_score)
    
    # Generate clinical report
    report = {
        'patient_id': patient_data['patient_id'],
        'analysis_date': pd.Timestamp.now(),
        'depression_risk': depression_risk,
        'construct_scores': {
            'behavioral_activation': ba_score,
            'social_engagement': se_score,
            'routine_stability': rs_score
        },
        'recommendations': generate_recommendations(depression_risk, construct_scores),
        'follow_up_needed': depression_risk['category'] in ['HIGH_RISK', 'MODERATE_RISK']
    }
    
    return report

def calculate_depression_risk(ba_score, se_score, rs_score):
    """Calculate depression risk from construct scores."""
    
    # Weighted risk algorithm
    risk_score = (-ba_score * 0.4) + (-se_score * 0.3) + (-rs_score * 0.3)
    
    if risk_score > 0.7:
        category = "HIGH_RISK"
        probability = 0.85
        recommendation = "Immediate clinical evaluation recommended"
    elif risk_score > 0.3:
        category = "MODERATE_RISK"
        probability = 0.65
        recommendation = "Schedule clinical assessment within 2 weeks"
    else:
        category = "LOW_RISK"
        probability = 0.25
        recommendation = "Continue routine monitoring"
    
    return {
        'category': category,
        'probability': probability,
        'recommendation': recommendation,
        'risk_score': risk_score
    }

# Usage in clinic
patient_data = load_patient_data('patient_001')
screening_report = depression_screening(patient_data)

if screening_report['follow_up_needed']:
    schedule_clinical_appointment(screening_report['patient_id'])
```

#### Treatment Monitoring

```python
class DepressionTreatmentMonitor:
    """Monitor depression treatment response using digital phenotyping."""
    
    def __init__(self, patient_id, baseline_data):
        self.patient_id = patient_id
        self.baseline_scores = self._analyze_baseline(baseline_data)
        self.treatment_start = pd.Timestamp.now()
    
    def _analyze_baseline(self, baseline_data):
        """Analyze baseline digital phenotyping data."""
        return analyze_participant(participant_id=self.patient_id, **baseline_data)
    
    def monitor_progress(self, current_data):
        """Monitor treatment progress with current data."""
        
        # Analyze current data
        current_scores = analyze_participant(participant_id=self.patient_id, **current_data)
        
        # Calculate changes from baseline
        changes = self._calculate_changes(current_scores)
        
        # Determine treatment response
        response = self._assess_response(changes)
        
        # Generate progress report
        report = {
            'patient_id': self.patient_id,
            'weeks_in_treatment': (pd.Timestamp.now() - self.treatment_start).days / 7,
            'baseline_scores': self._extract_scores(self.baseline_scores),
            'current_scores': self._extract_scores(current_scores),
            'changes': changes,
            'response_category': response['category'],
            'response_magnitude': response['magnitude'],
            'clinical_recommendations': response['recommendations']
        }
        
        return report
    
    def _calculate_changes(self, current_scores):
        """Calculate changes from baseline."""
        changes = {}
        
        for construct in ['behavioral_activation', 'social_engagement', 'routine_stability']:
            baseline = self.baseline_scores[construct].normalized_score
            current = current_scores[construct].normalized_score
            changes[construct] = current - baseline
        
        return changes
    
    def _assess_response(self, changes):
        """Assess treatment response based on score changes."""
        
        # Overall improvement score
        overall_change = sum(changes.values()) / len(changes)
        
        if overall_change > 0.5:
            category = "STRONG_RESPONSE"
            magnitude = "Large improvement"
            recommendations = ["Continue current treatment", "Consider maintenance planning"]
        elif overall_change > 0.2:
            category = "MODERATE_RESPONSE"
            magnitude = "Moderate improvement"
            recommendations = ["Continue current treatment", "Monitor progress"]
        elif overall_change > -0.2:
            category = "MINIMAL_RESPONSE"
            magnitude = "Minimal change"
            recommendations = ["Consider treatment adjustment", "Increase monitoring frequency"]
        else:
            category = "NO_RESPONSE"
            magnitude = "No improvement or worsening"
            recommendations = ["Re-evaluate treatment plan", "Consider alternative approaches"]
        
        return {
            'category': category,
            'magnitude': magnitude,
            'recommendations': recommendations
        }

# Usage in clinical practice
monitor = DepressionTreatmentMonitor('patient_001', baseline_data)
progress_report = monitor.monitor_progress(current_data)
```

### Anxiety Management

#### Social Anxiety Monitoring

```python
def social_anxiety_assessment(digital_data):
    """Assess social anxiety using digital phenotyping markers."""
    
    # Analyze constructs
    construct_scores = analyze_participant(**digital_data, participant_id=digital_data['participant_id'])
    
    # Extract anxiety-relevant markers
    avoidance_score = construct_scores['avoidance'].normalized_score
    se_score = construct_scores['social_engagement'].normalized_score
    
    # Calculate social anxiety indicators
    home_confinement = construct_scores['avoidance'].feature_scores.get('home_confinement', 0)
    communication_gaps = construct_scores['avoidance'].feature_scores.get('communication_gaps', 0)
    initiation_rate = construct_scores['social_engagement'].feature_scores.get('initiation_rate', 0)
    
    # Social anxiety risk assessment
    anxiety_risk = assess_social_anxiety_risk(
        avoidance_score, se_score, 
        home_confinement, communication_gaps, initiation_rate
    )
    
    return {
        'participant_id': digital_data['participant_id'],
        'social_anxiety_risk': anxiety_risk,
        'behavioral_patterns': {
            'avoidance_level': avoidance_score,
            'social_engagement': se_score,
            'home_confinement': home_confinement,
            'communication_gaps': communication_gaps,
            'initiation_rate': initiation_rate
        },
        'recommendations': generate_anxiety_recommendations(anxiety_risk)
    }

def assess_social_anxiety_risk(avoidance, social_engagement, home_confinement, comm_gaps, initiation):
    """Assess social anxiety risk from multiple markers."""
    
    risk_score = 0
    
    # High avoidance contributes to risk
    if avoidance > 0.5:
        risk_score += 0.3
    elif avoidance > 0.2:
        risk_score += 0.15
    
    # Low social engagement contributes to risk
    if social_engagement < -0.5:
        risk_score += 0.3
    elif social_engagement < -0.2:
        risk_score += 0.15
    
    # High home confinement contributes to risk
    if home_confinement > 0.6:
        risk_score += 0.2
    elif home_confinement > 0.4:
        risk_score += 0.1
    
    # Communication gaps contribute to risk
    if comm_gaps > 0.5:
        risk_score += 0.1
    
    # Low initiation rate contributes to risk
    if initiation < 0.3:
        risk_score += 0.1
    
    # Categorize risk
    if risk_score > 0.7:
        return "HIGH_RISK"
    elif risk_score > 0.4:
        return "MODERATE_RISK"
    else:
        return "LOW_RISK"
```

---

## Research Applications

### Population Studies

#### Cross-Sectional Analysis

```python
import pandas as pd
import numpy as np
from scipy import stats

class PopulationStudy:
    """Conduct population-level digital phenotyping studies."""
    
    def __init__(self, study_config):
        self.config = study_config
        self.results = {}
    
    def analyze_population(self, participant_data_dict):
        """Analyze digital phenotyping across population."""
        
        all_construct_scores = []
        participant_metadata = []
        
        # Analyze each participant
        for participant_id, data in participant_data_dict.items():
            try:
                construct_scores = analyze_participant(participant_id=participant_id, **data)
                scores_df = self._extract_scores_dataframe(construct_scores, participant_id)
                all_construct_scores.append(scores_df)
                
                # Collect metadata
                metadata = data.get('metadata', {})
                metadata['participant_id'] = participant_id
                participant_metadata.append(metadata)
                
            except Exception as e:
                print(f"Error analyzing participant {participant_id}: {e}")
                continue
        
        # Combine results
        combined_scores = pd.concat(all_construct_scores, ignore_index=True)
        metadata_df = pd.DataFrame(participant_metadata)
        
        # Conduct population analyses
        self.results = {
            'descriptive_stats': self._calculate_descriptive_stats(combined_scores),
            'correlations': self._calculate_correlations(combined_scores),
            'demographic_analyses': self._analyze_demographics(combined_scores, metadata_df),
            'cluster_analysis': self._conduct_cluster_analysis(combined_scores),
            'population_distributions': self._analyze_distributions(combined_scores)
        }
        
        return self.results
    
    def _extract_scores_dataframe(self, construct_scores, participant_id):
        """Extract construct scores into DataFrame."""
        
        data = {'participant_id': participant_id}
        
        for construct_name, construct_score in construct_scores.items():
            data[f'{construct_name}_score'] = construct_score.normalized_score
            data[f'{construct_name}_quality'] = construct_score.quality_metrics['overall_quality']
            
            # Add feature scores
            for feature_name, feature_score in construct_score.feature_scores.items():
                data[f'{construct_name}_{feature_name}'] = feature_score
        
        return pd.DataFrame([data])
    
    def _calculate_descriptive_stats(self, scores_df):
        """Calculate descriptive statistics for population."""
        
        construct_columns = [col for col in scores_df.columns if col.endswith('_score')]
        
        stats_dict = {}
        for col in construct_columns:
            construct_name = col.replace('_score', '')
            stats_dict[construct_name] = {
                'mean': scores_df[col].mean(),
                'std': scores_df[col].std(),
                'median': scores_df[col].median(),
                'min': scores_df[col].min(),
                'max': scores_df[col].max(),
                'q25': scores_df[col].quantile(0.25),
                'q75': scores_df[col].quantile(0.75),
                'skewness': stats.skew(scores_df[col]),
                'kurtosis': stats.kurtosis(scores_df[col]),
                'n_valid': scores_df[col].notna().sum()
            }
        
        return stats_dict
    
    def _calculate_correlations(self, scores_df):
        """Calculate correlations between constructs."""
        
        construct_columns = [col for col in scores_df.columns if col.endswith('_score')]
        correlation_matrix = scores_df[construct_columns].corr()
        
        # Calculate p-values
        p_values = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns)
        
        for col1 in construct_columns:
            for col2 in construct_columns:
                if col1 != col2:
                    _, p_val = stats.pearsonr(scores_df[col1].dropna(), scores_df[col2].dropna())
                    p_values.loc[col1, col2] = p_val
        
        return {
            'correlation_matrix': correlation_matrix,
            'p_values': p_values,
            'significant_correlations': self._find_significant_correlations(correlation_matrix, p_values)
        }
    
    def generate_report(self, output_file):
        """Generate comprehensive population study report."""
        
        report = f"""
# Digital Phenotyping Population Study Report

## Study Overview
- Total Participants: {len(self.results.get('descriptive_stats', {}).get('behavioral_activation', {}).get('n_valid', 0))}
- Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Descriptive Statistics
"""
        
        # Add descriptive statistics
        for construct, stats in self.results['descriptive_stats'].items():
            report += f"""
### {construct.replace('_', ' ').title()}
- Mean: {stats['mean']:.3f} (SD: {stats['std']:.3f})
- Median: {stats['median']:.3f}
- Range: [{stats['min']:.3f}, {stats['max']:.3f}]
- Skewness: {stats['skewness']:.3f}
- Valid Cases: {stats['n_valid']}
"""
        
        # Add correlations
        report += "\n## Construct Correlations\n"
        significant_corrs = self.results['correlations']['significant_correlations']
        if significant_corrs:
            for corr in significant_corrs:
                report += f"- {corr['construct1']} & {corr['construct2']}: r = {corr['correlation']:.3f}, p = {corr['p_value']:.3f}\n"
        else:
            report += "No significant correlations found.\n"
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        return report

# Usage example
study_config = {
    'min_data_quality': 0.5,
    'analysis_window_days': 7,
    'constructs': ['behavioral_activation', 'avoidance', 'social_engagement', 'routine_stability']
}

population_study = PopulationStudy(study_config)
results = population_study.analyze_population(participant_data_dict)
population_study.generate_report('population_study_report.md')
```

### Predictive Modeling

#### Depression Prediction Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

class DepressionPredictor:
    """Predict depression outcomes from digital phenotyping data."""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
    def train_model(self, digital_data, clinical_labels, validation_split=0.2):
        """Train depression prediction model."""
        
        # Extract features for all participants
        X, y = self._prepare_training_data(digital_data, clinical_labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Validate model
        val_predictions = self.model.predict(X_val)
        val_probabilities = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = np.mean(val_predictions == y_val)
        auc = roc_auc_score(y_val, val_probabilities)
        
        validation_results = {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': classification_report(y_val, val_predictions),
            'feature_importance': self._get_feature_importance()
        }
        
        return validation_results
    
    def _prepare_training_data(self, digital_data, clinical_labels):
        """Prepare training data from digital phenotyping."""
        
        X = []
        y = []
        participant_ids = []
        
        for participant_id, data in digital_data.items():
            try:
                # Extract construct scores
                construct_scores = analyze_participant(participant_id=participant_id, **data)
                
                # Create feature vector
                feature_vector = self._create_feature_vector(construct_scores)
                X.append(feature_vector)
                y.append(clinical_labels[participant_id])
                participant_ids.append(participant_id)
                
            except Exception as e:
                print(f"Error processing participant {participant_id}: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        # Store feature names
        self.feature_names = self._get_feature_names(construct_scores)
        
        return X, y
    
    def _create_feature_vector(self, construct_scores):
        """Create feature vector from construct scores."""
        
        features = []
        
        # Add construct scores
        for construct_name in ['behavioral_activation', 'avoidance', 'social_engagement', 'routine_stability']:
            if construct_name in construct_scores:
                features.append(construct_scores[construct_name].normalized_score)
                
                # Add quality metrics
                features.append(construct_scores[construct_name].quality_metrics['overall_quality'])
                
                # Add feature scores
                for feature_name, score in construct_scores[construct_name].feature_scores.items():
                    features.append(score)
            else:
                # Add zeros for missing constructs
                features.extend([0.0] * (2 + 5))  # score + quality + up to 5 features
        
        return features
    
    def predict_depression(self, digital_data):
        """Predict depression for new participants."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        predictions = {}
        
        for participant_id, data in digital_data.items():
            try:
                # Extract features
                construct_scores = analyze_participant(participant_id=participant_id, **data)
                feature_vector = self._create_feature_vector(construct_scores)
                
                # Make prediction
                prediction = self.model.predict([feature_vector])[0]
                probability = self.model.predict_proba([feature_vector])[0, 1]
                
                predictions[participant_id] = {
                    'prediction': prediction,
                    'probability': probability,
                    'construct_scores': construct_scores
                }
                
            except Exception as e:
                print(f"Error predicting for participant {participant_id}: {e}")
                continue
        
        return predictions
    
    def _get_feature_importance(self):
        """Get feature importance from trained model."""
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        else:
            return None

# Usage example
predictor = DepressionPredictor()
validation_results = predictor.train_model(digital_data, clinical_labels)

# Make predictions on new data
new_predictions = predictor.predict_depression(new_digital_data)
```

---

## Advanced Configurations

### Custom Normalization

```python
from psyconstruct.constructs import AggregationConfig

# Custom configuration for clinical research
clinical_config = AggregationConfig(
    normalization_method="zscore",
    within_participant=False,  # Use population normalization
    aggregation_method="weighted_mean",
    handle_missing="mean_impute",
    min_features_required=3,
    min_quality_threshold=0.7,
    include_feature_scores=True,
    include_quality_metrics=True,
    include_normalization_params=True
)

# Custom configuration for individual monitoring
monitoring_config = AggregationConfig(
    normalization_method="zscore",
    within_participant=True,  # Use within-participant normalization
    aggregation_method="weighted_mean",
    handle_missing="exclude",
    min_features_required=2,
    min_quality_threshold=0.5,
    include_feature_scores=True,
    include_quality_metrics=True,
    include_normalization_params=True
)

# Use custom configuration
aggregator = ConstructAggregator(config=clinical_config)
results = aggregator.aggregate_all_constructs(feature_results, participant_id='patient_001')
```

### Reference Data Management

```python
def create_reference_data(population_data):
    """Create reference data for population normalization."""
    
    reference_data = {}
    
    # Analyze all participants in population
    all_construct_scores = {}
    
    for participant_id, data in population_data.items():
        construct_scores = analyze_participant(participant_id=participant_id, **data)
        
        for construct_name, construct_score in construct_scores.items():
            if construct_name not in all_construct_scores:
                all_construct_scores[construct_name] = {}
            
            for feature_name, feature_score in construct_score.feature_scores.items():
                if feature_name not in all_construct_scores[construct_name]:
                    all_construct_scores[construct_name][feature_name] = []
                all_construct_scores[construct_name][feature_name].append(feature_score)
    
    # Calculate reference statistics
    for construct_name, features in all_construct_scores.items():
        reference_data[construct_name] = {}
        
        for feature_name, scores in features.items():
            scores_array = np.array(scores)
            reference_data[construct_name][feature_name] = {
                'mean': np.mean(scores_array),
                'std': np.std(scores_array),
                'median': np.median(scores_array),
                'mad': np.median(np.abs(scores_array - np.median(scores_array))),
                'min': np.min(scores_array),
                'max': np.max(scores_array),
                'count': len(scores_array)
            }
    
    return reference_data

# Usage
reference_data = create_reference_data(population_data)

# Use reference data for normalization
results = aggregator.aggregate_all_constructs(
    feature_results, 
    participant_id='new_participant',
    reference_data=reference_data
)
```

---

## Integration Examples

### Electronic Health Record Integration

```python
class EHRIntegration:
    """Integrate digital phenotyping with electronic health records."""
    
    def __init__(self, ehr_connection, digital_phenotyping_system):
        self.ehr = ehr_connection
        self.dps = digital_phenotyping_system
        self.patient_cache = {}
    
    def update_patient_record(self, patient_id, digital_data):
        """Update patient record with digital phenotyping results."""
        
        # Analyze digital data
        construct_scores = self.dps.analyze_participant(
            participant_id=patient_id, 
            **digital_data
        )
        
        # Create clinical summary
        clinical_summary = self._create_clinical_summary(construct_scores)
        
        # Update EHR
        ehr_update = {
            'patient_id': patient_id,
            'digital_phenotyping': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'construct_scores': self._serialize_scores(construct_scores),
                'clinical_summary': clinical_summary,
                'data_quality': self._assess_data_quality(construct_scores)
            }
        }
        
        # Send to EHR
        self.ehr.update_patient_record(ehr_update)
        
        # Check for alerts
        alerts = self._generate_alerts(construct_scores)
        if alerts:
            self.ehr.create_clinical_alerts(patient_id, alerts)
        
        return clinical_summary
    
    def _create_clinical_summary(self, construct_scores):
        """Create clinical summary of digital phenotyping results."""
        
        summary = {
            'overall_status': 'stable',
            'concerns': [],
            'recommendations': [],
            'trend': 'stable'
        }
        
        # Assess each construct
        for construct_name, construct_score in construct_scores.items():
            score = construct_score.normalized_score
            interpretation = construct_score.interpretation
            
            if 'concern' in interpretation.lower() or 'elevated' in interpretation.lower():
                summary['concerns'].append(f"{construct_name}: {interpretation}")
                summary['overall_status'] = 'attention_needed'
            
            if score < -1.0:  # Significantly low
                summary['recommendations'].append(f"Monitor {construct_name} closely")
        
        return summary
    
    def _generate_alerts(self, construct_scores):
        """Generate clinical alerts based on construct scores."""
        
        alerts = []
        
        # High depression risk
        ba_score = construct_scores.get('behavioral_activation', {}).normalized_score
        se_score = construct_scores.get('social_engagement', {}).normalized_score
        
        if ba_score < -1.5 and se_score < -1.0:
            alerts.append({
                'type': 'depression_risk',
                'severity': 'high',
                'message': 'High depression risk detected. Immediate clinical evaluation recommended.',
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        # High anxiety risk
        av_score = construct_scores.get('avoidance', {}).normalized_score
        
        if av_score > 1.5:
            alerts.append({
                'type': 'anxiety_risk',
                'severity': 'moderate',
                'message': 'Elevated anxiety markers detected. Consider anxiety assessment.',
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        # Poor data quality
        overall_quality = np.mean([
            score.quality_metrics['overall_quality'] 
            for score in construct_scores.values()
        ])
        
        if overall_quality < 0.5:
            alerts.append({
                'type': 'data_quality',
                'severity': 'low',
                'message': 'Poor data quality detected. Check device compliance.',
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        return alerts

# Usage
ehr_integration = EHRIntegration(ehr_system, psyconstruct_system)
clinical_summary = ehr_integration.update_patient_record('patient_001', digital_data)
```

### Mobile App Integration

```python
class MobileAppIntegration:
    """Integrate digital phenotyping with mobile health applications."""
    
    def __init__(self, api_endpoint, authentication_token):
        self.api_endpoint = api_endpoint
        self.auth_token = authentication_token
    
    def send_real_time_feedback(self, participant_id, construct_scores):
        """Send real-time feedback to mobile app."""
        
        feedback = {
            'participant_id': participant_id,
            'timestamp': pd.Timestamp.now().isoformat(),
            'construct_scores': self._prepare_mobile_scores(construct_scores),
            'insights': self._generate_insights(construct_scores),
            'recommendations': self._generate_mobile_recommendations(construct_scores)
        }
        
        # Send to mobile app
        response = self._send_to_app('feedback', feedback)
        return response
    
    def _prepare_mobile_scores(self, construct_scores):
        """Prepare scores for mobile app display."""
        
        mobile_scores = {}
        
        for construct_name, construct_score in construct_scores.items():
            mobile_scores[construct_name] = {
                'score': construct_score.normalized_score,
                'status': self._get_status_indicator(construct_score.normalized_score),
                'trend': self._calculate_trend(construct_score),  # If historical data available
                'interpretation': construct_score.interpretation
            }
        
        return mobile_scores
    
    def _get_status_indicator(self, score):
        """Get status indicator for mobile display."""
        
        if score > 0.5:
            return 'excellent'
        elif score > 0.0:
            return 'good'
        elif score > -0.5:
            return 'concerning'
        else:
            return 'attention_needed'
    
    def _generate_insights(self, construct_scores):
        """Generate personalized insights for mobile app."""
        
        insights = []
        
        # Behavioral activation insights
        ba_score = construct_scores.get('behavioral_activation', {}).normalized_score
        if ba_score < -0.5:
            insights.append("Your activity levels have been lower than usual. Consider scheduling enjoyable activities.")
        
        # Social engagement insights
        se_score = construct_scores.get('social_engagement', {}).normalized_score
        if se_score < -0.5:
            insights.append("You've been less socially connected recently. Consider reaching out to friends or family.")
        
        # Routine stability insights
        rs_score = construct_scores.get('routine_stability', {}).normalized_score
        if rs_score < -0.5:
            insights.append("Your daily routine has been irregular. Maintaining consistent sleep and activity times can help.")
        
        return insights
    
    def _generate_mobile_recommendations(self, construct_scores):
        """Generate actionable recommendations for mobile app."""
        
        recommendations = []
        
        for construct_name, construct_score in construct_scores.items():
            score = construct_score.normalized_score
            
            if construct_name == 'behavioral_activation' and score < -0.5:
                recommendations.append({
                    'type': 'activity',
                    'title': 'Increase Physical Activity',
                    'description': 'Try to take a 10-minute walk today',
                    'priority': 'high'
                })
            
            elif construct_name == 'social_engagement' and score < -0.5:
                recommendations.append({
                    'type': 'social',
                    'title': 'Connect with Others',
                    'description': 'Send a message to a friend or family member',
                    'priority': 'medium'
                })
            
            elif construct_name == 'routine_stability' and score < -0.5:
                recommendations.append({
                    'type': 'routine',
                    'title': 'Stabilize Sleep Schedule',
                    'description': 'Try to go to bed and wake up at consistent times',
                    'priority': 'medium'
                })
        
        return recommendations

# Usage
mobile_integration = MobileAppIntegration('https://api.healthapp.com', 'auth_token')
feedback_response = mobile_integration.send_real_time_feedback('user_001', construct_scores)
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Data Quality Problems

```python
def diagnose_data_quality(data_source, data):
    """Diagnose common data quality issues."""
    
    diagnosis = {
        'issues': [],
        'recommendations': [],
        'quality_score': 0.0
    }
    
    # Check data completeness
    if 'timestamp' in data:
        expected_points = calculate_expected_points(data['timestamp'])
        actual_points = len(data['timestamp'])
        completeness = actual_points / expected_points
        
        if completeness < 0.7:
            diagnosis['issues'].append(f"Low data completeness: {completeness:.1%}")
            diagnosis['recommendations'].append("Check device battery and data collection settings")
    
    # Check sampling rate
    if len(data['timestamp']) > 1:
        time_diffs = np.diff(data['timestamp']).astype('timedelta64[s]').astype(float)
        median_interval = np.median(time_diffs)
        
        if median_interval > 300:  # 5 minutes
            diagnosis['issues'].append(f"Low sampling rate: {median_interval:.0f} seconds")
            diagnosis['recommendations'].append("Increase sensor sampling frequency")
    
    # Check for outliers
    for key in ['x', 'y', 'z'] if key in data else []:
        if key in data:
            values = np.array(data[key])
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            outliers = np.sum((values > q75 + 1.5*iqr) | (values < q25 - 1.5*iqr))
            
            if outliers / len(values) > 0.1:
                diagnosis['issues'].append(f"High outlier ratio in {key}: {outliers/len(values):.1%}")
                diagnosis['recommendations'].append("Check sensor calibration and placement")
    
    # Calculate overall quality score
    diagnosis['quality_score'] = max(0.0, 1.0 - len(diagnosis['issues']) * 0.2)
    
    return diagnosis

# Usage
quality_diagnosis = diagnose_data_quality('accelerometer', accelerometer_data)
```

#### Feature Extraction Errors

```python
def troubleshoot_feature_extraction(error, data_type, data):
    """Troubleshoot feature extraction errors."""
    
    troubleshooting = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'likely_causes': [],
        'solutions': []
    }
    
    if "insufficient data" in str(error).lower():
        troubleshooting['likely_causes'].append("Insufficient data coverage")
        troubleshooting['solutions'].append("Ensure minimum data collection requirements are met")
        troubleshooting['solutions'].append("Check data collection period and coverage")
    
    elif "invalid format" in str(error).lower():
        troubleshooting['likely_causes'].append("Incorrect data format")
        troubleshooting['solutions'].append("Verify data format matches requirements")
        troubleshooting['solutions'].append("Check column names and data types")
    
    elif "missing required" in str(error).lower():
        troubleshooting['likely_causes'].append("Missing required columns")
        troubleshooting['solutions'].append("Ensure all required columns are present")
        troubleshooting['solutions'].append("Check data preprocessing pipeline")
    
    elif "quality threshold" in str(error).lower():
        troubleshooting['likely_causes'].append("Data quality below threshold")
        troubleshooting['solutions'].append("Improve data collection conditions")
        troubleshooting['solutions'].append("Consider lowering quality threshold for testing")
    
    return troubleshooting

# Usage
try:
    result = ba_features.activity_volume(accelerometer_data)
except Exception as e:
    troubleshooting = troubleshoot_feature_extraction(e, 'accelerometer', accelerometer_data)
    print(f"Error: {troubleshooting['error_type']}")
    print(f"Causes: {troubleshooting['likely_causes']}")
    print(f"Solutions: {troubleshooting['solutions']}")
```

#### Performance Optimization

```python
def optimize_performance(data_size, available_memory):
    """Optimize performance based on data characteristics."""
    
    optimizations = []
    
    # Memory optimization
    if data_size > available_memory * 0.8:
        optimizations.append("Use data chunking for large datasets")
        optimizations.append("Consider data sampling for initial analysis")
    
    # Processing optimization
    if data_size > 1000000:  # Large dataset
        optimizations.append("Enable parallel processing")
        optimizations.append("Use optimized data structures")
    
    # I/O optimization
    if data_size > 10000000:  # Very large dataset
        optimizations.append("Use efficient file formats (Parquet, HDF5)")
        optimizations.append("Implement data streaming")
    
    return optimizations

# Usage
optimizations = optimize_performance(len(accelerometer_data['timestamp']), available_memory_gb * 1024**3)
```

This comprehensive practical examples guide provides detailed implementations for clinical, research, and integration scenarios, along with troubleshooting guidance for common issues.
