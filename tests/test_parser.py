import pytest
from unittest.mock import MagicMock, patch
from src.services.parser_service import ParserService

@pytest.fixture
def parser_service():
    # Force api_enabled to False to test local parser
    with patch('src.services.parser_service.settings') as mock_settings:
        mock_settings.GEMINI_API_KEY = ""
        service = ParserService()
        yield service

def test_parse_with_heuristics_basic(parser_service):
    """Tests if local heuristic parser correctly extracts basic details from clean text."""
    resume_text = """
    John Doe
    john.doe@example.com | (123) 456-7890
    
    Education
    Bachelor of Science in Computer Science
    University of Tech, 2020 - 2024
    
    Experience
    Software Engineer at Tech Corp (2022 - Present)
    - Developed web applications using Python and Django.
    - Wrote SQL queries and managed PostgreSQL databases.
    
    Skills
    Python, SQL, Django, Git, Docker
    """
    
    parsed = parser_service._parse_with_heuristics(resume_text)
    
    assert parsed["candidate_name"] == "John Doe"
    assert parsed["email"] == "john.doe@example.com"
    assert parsed["phone"] == "(123) 456-7890"
    assert "python" in parsed["skills"]
    assert "sql" in parsed["skills"]
    assert "docker" in parsed["skills"]
    
    # Verify education and experience arrays are populated
    assert len(parsed["education"]) > 0
    assert len(parsed["experience"]) > 0

def test_parse_with_heuristics_edge_cases(parser_service):
    """Tests parser with missing or messy info."""
    messy_text = "No contact info here. Only skills like Java and Kubernetes."
    parsed = parser_service._parse_with_heuristics(messy_text)
    
    assert parsed["email"] is None
    assert parsed["phone"] is None
    assert "java" in parsed["skills"]
    assert "kubernetes" in parsed["skills"]

@patch("os.path.exists")
@patch("fitz.open")
def test_extract_text_from_pdf(mock_fitz_open, mock_exists, parser_service):
    """Tests if PDF text extraction calls PyMuPDF methods correctly."""
    mock_exists.return_value = True
    # Setup mock page and document
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Extracted PDF Text"
    
    mock_doc = MagicMock()
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz_open.return_value = mock_doc
    
    text = parser_service.extract_text_from_pdf("dummy_path.pdf")
    
    assert text == "Extracted PDF Text"
    mock_fitz_open.assert_called_once_with("dummy_path.pdf")
    mock_page.get_text.assert_called_once()
    mock_doc.close.assert_called_once()
