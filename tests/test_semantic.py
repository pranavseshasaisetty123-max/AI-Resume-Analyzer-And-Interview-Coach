import pytest
from unittest.mock import patch
from src.services.semantic_service import SemanticService

@pytest.fixture
def semantic_service():
    service = SemanticService()
    # Force is_transformer_loaded to False so it uses fallbacks in tests
    service.is_transformer_loaded = False
    return service

def test_tfidf_similarity_identical(semantic_service):
    """Identical texts should have 1.0 or very high similarity."""
    text = "Python software engineer with experience building web applications in FastAPI and Docker."
    score = semantic_service._calculate_tfidf_similarity(text, text)
    assert score == pytest.approx(1.0, 0.01)

def test_tfidf_similarity_disjoint(semantic_service):
    """Completely unrelated texts should have 0.0 similarity."""
    text1 = "Python software engineer building web applications"
    text2 = "Looking for a nurse or healthcare worker in local clinic"
    score = semantic_service._calculate_tfidf_similarity(text1, text2)
    assert score == pytest.approx(0.0, 0.01)

def test_tfidf_similarity_overlap(semantic_service):
    """Partially overlapping texts should have moderate similarity."""
    text1 = "Python developer with database experience in SQL and PostgreSQL."
    text2 = "Looking for a software developer who knows Python, Java, and SQL databases."
    score = semantic_service._calculate_tfidf_similarity(text1, text2)
    assert 0.1 < score < 0.9

@patch.object(SemanticService, "_load_transformer_model_lazy")
@patch.object(SemanticService, "_calculate_similarity_with_gemini")
def test_calculate_similarity_gemini_fallback(mock_gemini_sim, mock_lazy_load, semantic_service):
    """Verifies that if transformer is unavailable but Gemini is configured, it falls back to Gemini."""
    mock_lazy_load.return_value = False
    mock_gemini_sim.return_value = 0.85
    
    with patch('src.services.semantic_service.settings') as mock_settings:
        mock_settings.GEMINI_API_KEY = "dummy_key"
        
        score = semantic_service.calculate_similarity("Resume text", "Job Description")
        
        assert score == 0.85
        mock_gemini_sim.assert_called_once()
