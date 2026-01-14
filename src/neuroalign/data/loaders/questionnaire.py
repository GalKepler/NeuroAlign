"""
Questionnaire Data Loader for NeuroAlign
=========================================

Save this file as: src/neuroalign/data/loaders/questionnaire.py

Loads and processes behavioral/cognitive questionnaire data.

Example:
    >>> from neuroalign.data.loaders import QuestionnaireLoader
    >>> loader = QuestionnaireLoader("Qcenter_-_IntegratedQ.csv")
    >>> df = loader.load()
    >>> stats = loader.get_cohort_statistics(subject_codes=[...])
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


class QuestionnaireLoader:
    """
    Loader for questionnaire/behavioral data.
    
    Handles the IntegratedQ format with multiple assessment domains.
    """
    
    # Define feature categories
    DEMOGRAPHIC_FEATURES = [
        "Gender", "Age", "DominantHand", "Weight (kg)", "Height (cm)",
        "Gender Indentity", "Sexual Orientation", "Country of Birth",
        "Marital Status", "Number of Children"
    ]
    
    MENTAL_HEALTH_FEATURES = [
        "OASIS", "PCL-5", "PHQ9", "GAD7",
        "Depression", "Anxiety", "AttentionDisorders",
        "SubjectiveHappiness", "SWLS"
    ]
    
    PERSONALITY_FEATURES = [
        "B5 Extraversion", "B5 Agreeableness", "B5 Coscientioness",
        "B5 EmotionalStability", "B5 Openness"
    ]
    
    LIFESTYLE_FEATURES = [
        "HoobyTime", "TimesTrainingPerWeek", "Caffeine", "Water",
        "SugarBeverages", "Alcohol", "Smoking", "Canabis",
        "ScreenTime", "PSQI"
    ]
    
    SOCIOECONOMIC_FEATURES = [
        "Education", "WorkStatus", "Salary", "PsychometricScore"
    ]
    
    def __init__(self, questionnaire_path: Path):
        """
        Initialize questionnaire loader.
        
        Args:
            questionnaire_path: Path to IntegratedQ CSV file
        """
        self.questionnaire_path = Path(questionnaire_path)
        self.data: Optional[pd.DataFrame] = None
    
    def load(
        self,
        clean: bool = True,
        standardize_subject_codes: bool = True
    ) -> pd.DataFrame:
        """
        Load questionnaire data from CSV.
        
        Args:
            clean: Whether to apply basic cleaning
            standardize_subject_codes: Whether to standardize subject code format
            
        Returns:
            DataFrame with questionnaire responses
        """
        self.data = pd.read_csv(self.questionnaire_path)
        
        if standardize_subject_codes:
            self.data["Subject Code"] = (
                self.data["Subject Code"]
                .astype(str)
                .str.replace("-", "")
                .str.replace("_", "")
                .str.replace(" ", "")
            )
        
        if clean:
            # Remove rows where Questionnaire is "No"
            if "Questionnaire" in self.data.columns:
                self.data = self.data[self.data["Questionnaire"] != "No"]
            
            # Convert numeric columns
            for col in self.MENTAL_HEALTH_FEATURES + ["Age", "Weight (kg)", "Height (cm)"]:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
        
        return self.data
    
    def merge_with_sessions(
        self,
        sessions_df: pd.DataFrame,
        on: str = "Subject Code",
        subject_col: str = "subject_code"
    ) -> pd.DataFrame:
        """
        Merge questionnaire data with session DataFrame.
        
        Args:
            sessions_df: DataFrame with session information
            on: Column name in questionnaire data to merge on
            subject_col: Column name in sessions_df for subject codes
            
        Returns:
            Merged DataFrame
        """
        if self.data is None:
            raise ValueError("Must call load() before merging")
        
        # Standardize subject codes in sessions_df
        sessions_copy = sessions_df.copy()
        sessions_copy[subject_col] = (
            sessions_copy[subject_col]
            .astype(str)
            .str.replace("-", "")
            .str.replace("_", "")
            .str.replace(" ", "")
        )
        
        merged = sessions_copy.merge(
            self.data,
            left_on=subject_col,
            right_on=on,
            how="left"
        )
        
        return merged
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get features organized by category.
        
        Returns:
            Dictionary mapping category names to lists of feature columns
        """
        if self.data is None:
            raise ValueError("Must call load() first")
        
        available_features = {}
        
        for category, features in [
            ("demographics", self.DEMOGRAPHIC_FEATURES),
            ("mental_health", self.MENTAL_HEALTH_FEATURES),
            ("personality", self.PERSONALITY_FEATURES),
            ("lifestyle", self.LIFESTYLE_FEATURES),
            ("socioeconomic", self.SOCIOECONOMIC_FEATURES)
        ]:
            available = [f for f in features if f in self.data.columns]
            if available:
                available_features[category] = available
        
        return available_features
    
    def get_numeric_features(self) -> List[str]:
        """Get list of numeric feature columns."""
        if self.data is None:
            raise ValueError("Must call load() first")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["UID", "EpigeneticScore", "AdjustedEpigeneticScore"]
        return [col for col in numeric_cols if col not in exclude]
    
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature columns."""
        if self.data is None:
            raise ValueError("Must call load() first")
        
        categorical_cols = self.data.select_dtypes(include=["object"]).columns.tolist()
        exclude = ["Subject Code", "UID", "Questionnaire", "LAB"]
        return [col for col in categorical_cols if col not in exclude]
    
    def summarize_participant(
        self,
        subject_code: str,
        categories: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Get summary of a participant's questionnaire responses.
        
        Args:
            subject_code: Subject identifier
            categories: List of category names to include
            
        Returns:
            Dictionary with participant's questionnaire data
        """
        if self.data is None:
            raise ValueError("Must call load() first")
        
        # Standardize subject code
        subject_code = str(subject_code).replace("-", "").replace("_", "").replace(" ", "")
        
        participant = self.data[self.data["Subject Code"] == subject_code]
        if len(participant) == 0:
            return {}
        
        participant = participant.iloc[0]
        
        feature_groups = self.get_feature_groups()
        if categories is None:
            categories = list(feature_groups.keys())
        
        summary = {}
        for category in categories:
            if category not in feature_groups:
                continue
            
            category_data = {}
            for feature in feature_groups[category]:
                value = participant.get(feature)
                if pd.notna(value):
                    category_data[feature] = value
            
            if category_data:
                summary[category] = category_data
        
        return summary
    
    def get_cohort_statistics(
        self,
        subject_codes: List[str],
        features: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get aggregate statistics for a cohort of participants.
        
        Args:
            subject_codes: List of subject identifiers
            features: List of features to summarize
            
        Returns:
            Dictionary with mean/std/median for each feature
        """
        if self.data is None:
            raise ValueError("Must call load() first")
        
        # Standardize subject codes
        subject_codes = [
            str(s).replace("-", "").replace("_", "").replace(" ", "")
            for s in subject_codes
        ]
        
        cohort = self.data[self.data["Subject Code"].isin(subject_codes)]
        
        if features is None:
            features = self.get_numeric_features()
        
        statistics = {}
        for feature in features:
            if feature not in cohort.columns:
                continue
            
            values = cohort[feature].dropna()
            if len(values) == 0:
                continue
            
            statistics[feature] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "n": len(values)
            }
        
        return statistics
