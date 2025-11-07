"""
Query Analysis and Intent Classification for MapleCLI
Analyzes user queries to determine intent and extract entities.
"""

import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("maplecli.query_analyzer")


class QueryAnalyzer:
    """
    Analyze user queries to determine intent and extract entities.
    """
    
    INTENT_PATTERNS = {
        'find_function': [
            r'where is (?:the )?(\w+) function',
            r'find (?:the )?(\w+) function',
            r'show me (?:the )?(\w+) function',
            r'locate (?:the )?(\w+) function',
        ],
        'find_class': [
            r'where is (?:the )?(\w+) class',
            r'find (?:the )?(\w+) class',
            r'show me (?:the )?(\w+) class',
            r'locate (?:the )?(\w+) class',
        ],
        'find_usages': [
            r'where is (\w+) used',
            r'find usages of (\w+)',
            r'who calls (\w+)',
            r'what calls (\w+)',
            r'find references to (\w+)',
        ],
        'explain_code': [
            r'explain (\w+)',
            r'what does (\w+) do',
            r'how does (\w+) work',
            r'describe (\w+)',
            r'tell me about (\w+)',
        ],
        'find_bugs': [
            r'find bugs',
            r'security issues',
            r'vulnerabilities',
            r'code smells',
            r'potential problems',
            r'review.*security',
        ],
        'architecture': [
            r'architecture',
            r'design patterns',
            r'structure',
            r'how is.*organized',
            r'project structure',
        ],
        'find_api': [
            r'api endpoints',
            r'find.*endpoints',
            r'list.*routes',
            r'show.*api',
        ],
        'dependencies': [
            r'dependencies',
            r'what.*depends on',
            r'imports',
            r'requirements',
        ],
        'refactor': [
            r'refactor',
            r'improve',
            r'optimize',
            r'better way',
        ],
        'test': [
            r'test',
            r'unit test',
            r'how to test',
        ]
    }
    
    def analyze(self, query: str) -> Dict:
        """Classify query intent and extract entities."""
        query_lower = query.lower()
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    entities = [e for e in match.groups() if e] if match.groups() else []
                    return {
                        'intent': intent,
                        'entities': entities,
                        'original_query': query,
                        'confidence': 'high'
                    }
        
        # Check for file-specific queries
        if self._is_file_query(query):
            return {
                'intent': 'file_specific',
                'entities': self._extract_filenames(query),
                'original_query': query,
                'confidence': 'medium'
            }
        
        return {
            'intent': 'general',
            'entities': [],
            'original_query': query,
            'confidence': 'low'
        }
    
    def _is_file_query(self, query: str) -> bool:
        """Check if query mentions specific files."""
        file_patterns = [
            r'\w+\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php)',
            r'in (?:the )?file',
            r'in (?:the )?(\w+/)+\w+',
        ]
        
        for pattern in file_patterns:
            if re.search(pattern, query.lower()):
                return True
        return False
    
    def _extract_filenames(self, query: str) -> List[str]:
        """Extract filenames from query."""
        filenames = []
        
        # Match common file extensions
        file_pattern = r'(\w+(?:/\w+)*\.\w+)'
        matches = re.findall(file_pattern, query)
        filenames.extend(matches)
        
        return filenames
    
    def get_search_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query for search."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'can', 'may', 'might', 'must', 'what', 'where', 'when', 'why',
            'how', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def should_use_semantic_search(self, analysis: Dict) -> bool:
        """Determine if semantic search should be used."""
        # Use semantic search for general queries and explanations
        semantic_intents = ['general', 'explain_code', 'architecture', 'find_bugs', 'refactor']
        return analysis['intent'] in semantic_intents
    
    def should_use_symbol_search(self, analysis: Dict) -> bool:
        """Determine if symbol search should be used."""
        # Use symbol search for specific lookups
        symbol_intents = ['find_function', 'find_class', 'find_usages']
        return analysis['intent'] in symbol_intents
    
    def format_context_prompt(self, query: str, context: str, analysis: Dict) -> str:
        """Format the final prompt with context based on intent."""
        intent = analysis['intent']
        
        if intent == 'find_function' or intent == 'find_class':
            return f"{query}\n\n[DEFINITION]:\n{context}"
        
        elif intent == 'find_usages':
            return f"{query}\n\n[USAGES]:\n{context}"
        
        elif intent == 'explain_code':
            return f"{query}\n\n[RELEVANT CODE]:\n{context}"
        
        elif intent == 'architecture':
            return f"{query}\n\n[PROJECT STRUCTURE]:\n{context}"
        
        elif intent == 'find_bugs':
            return f"{query}\n\n[CODE TO REVIEW]:\n{context}\n\nPlease analyze for bugs, security issues, and code quality problems."
        
        else:
            return f"{query}\n\n[RELEVANT CONTEXT]:\n{context}"

