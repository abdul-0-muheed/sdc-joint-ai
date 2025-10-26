import json
import os
import re
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union, Callable

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Handle imports for different LangChain versions
try:
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.schema import AttributeInfo
    SELF_QUERY_AVAILABLE = True
    print("✅ Using langchain.retrievers.self_query")
except ImportError:
    try:
        from langchain_community.retrievers.self_query.base import SelfQueryRetriever
        from langchain_community.chains.query_constructor.schema import AttributeInfo
        SELF_QUERY_AVAILABLE = True
        print("✅ Using langchain_community.retrievers.self_query")
    except ImportError:
        SELF_QUERY_AVAILABLE = False
        SelfQueryRetriever = None  # Define as None for type hints
        AttributeInfo = None  # Define as None
        print("⚠️ SelfQueryRetriever not available, using basic search")

# Paths
DATASET_FILE = r"data\restructured_sdit_dataset_auto_subcat.jsonl"
INDEX_PATH = "data/faiss_index_optimized"

class OptimizedRAGSystem:
    """Optimized RAG system for Shree Devi College information"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.retriever = None
        self.query_cache = {}
        self.llm = None
        
    def initialize_local_llm(self):
        """Initialize a local LLM for self-query construction"""
        try:
            # Try to use a smaller model that can run on CPU
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            # Check if we have enough resources
            if torch.cuda.is_available():
                device = "cuda"
                print("✅ CUDA GPU detected, using GPU acceleration")
            else:
                device = "cpu"
                print("⚠️ No GPU detected, using CPU (this may be slower)")
            
            # Create tokenizer and model
            print(f"🔄 Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                trust_remote_code=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                device=device
            )
            
            # Create LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            print("✅ Local LLM initialized successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not initialize local LLM: {e}")
            print("🔄 Falling back to basic retrieval without self-query capabilities")
            self.llm = None
            return False
    
    def load_and_validate_data(self, dataset_file: str) -> List[Dict[str, Any]]:
        """Load and validate dataset with improved error handling"""
        
        if not os.path.exists(dataset_file):
            print(f"❌ Dataset file not found: {dataset_file}")
            return []
        
        valid_data = []
        
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON parsing error on line {line_number}: {e}")
                        continue
                    
                    # Validate required fields
                    required_fields = ["content", "title", "category", "source"]
                    if not all(field in item for field in required_fields):
                        print(f"❌ Missing required fields on line {line_number}")
                        continue
                    
                    valid_data.append(item)
                    
        except Exception as e:
            print(f"❌ Error reading dataset file: {e}")
            return []
        
        print(f"✅ Found {len(valid_data)} valid entries in dataset")
        return valid_data
    
    def create_semantic_chunker(self) -> RecursiveCharacterTextSplitter:
        """Create a semantic-aware chunker for educational content"""
        
        # Define custom separators that make sense for educational content
        separators = [
            "\n\n## ",  # Markdown headers
            "\n\n### ", 
            "\n\n#### ",
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamation endings
            "? ",    # Question endings
            "; ",    # Semicolon
            ", ",    # Commas
            " ",     # Spaces
            ""       # Character level
        ]
        
        # Create chunker with semantic awareness
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for better precision
            chunk_overlap=150,  # Higher overlap for context preservation
            separators=separators,
            length_function=len,
        )
        
        return chunker
    
    def create_enhanced_documents(self, valid_data: List[Dict[str, Any]]) -> List[Document]:
        """Create documents with enhanced metadata and structure"""
        
        docs = []
        
        for item in valid_data:
            # Enhanced content with semantic markers
            content = item['content']
            
            # Add structured information
            enhanced_content = f"""
[CATEGORY: {item['category']}]
[SUBCATEGORY: {item.get('subcategory', 'General')}]
[TITLE: {item['title']}]
[CONTENT: {content}]
"""
            
            # Create rich metadata
            metadata = {
                "title": item["title"],
                "category": item["category"],
                "subcategory": item.get("subcategory", "General"),
                "source": item["source"],
                "content_type": "templated" if item.get("source") == "generated_template" else "factual",
                "has_contact_info": bool(re.search(r'\+?\d[\d\s-]{8,}\d', content)),
                "has_location_info": bool(re.search(r'(campus|address|location|road)', content.lower())),
                "has_course_info": bool(re.search(r'(course|program|degree|diploma)', content.lower())),
                "has_event_info": bool(re.search(r'(event|festival|competition|activity)', content.lower())),
                "has_admission_info": bool(re.search(r'(admission|apply|application|fee)', content.lower())),
            }
            
            docs.append(Document(page_content=enhanced_content, metadata=metadata))
        
        return docs
    
    def enhance_document_processing(self, docs: List[Document]) -> List[Document]:
        """Further enhance documents with additional processing"""
        
        enhanced_docs = []
        
        for doc in docs:
            # Extract key information from content
            content = doc.page_content
            
            # Add content length metadata
            doc.metadata["content_length"] = len(content)
            
            # Detect and mark contact information
            if re.search(r'\+?\d[\d\s-]{8,}\d', content):
                doc.metadata["contact_info_present"] = True
                # Extract phone numbers
                phone_numbers = re.findall(r'\+?\d[\d\s-]{8,}\d', content)
                doc.metadata["phone_numbers"] = phone_numbers
            
            # Detect and mark email addresses
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            if emails:
                doc.metadata["emails_present"] = True
                doc.metadata["emails"] = emails
            
            # Detect and mark addresses
            address_patterns = [
                r'\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{6}',  # Indian address pattern
                r'[A-Za-z\s]+Road,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+'  # Road pattern
            ]
            
            for pattern in address_patterns:
                if re.search(pattern, content):
                    doc.metadata["address_present"] = True
                    break
            
            enhanced_docs.append(doc)
        
        return enhanced_docs
    
    def create_self_query_retriever(self, vectorstore: FAISS) -> Optional[Any]:
        """Create a self-query retriever with metadata filtering"""
        
        if not SELF_QUERY_AVAILABLE:
            print("⚠️ SelfQueryRetriever not available in this LangChain version")
            return None
            
        if self.llm is None:
            print("⚠️ No LLM available for self-query retriever")
            return None
        
        # Define metadata field info
        metadata_field_info = [
            {
                "name": "category",
                "description": "The main category of the content (e.g., 'College Information', 'Departments & Courses', 'Admissions & Academics')",
                "type": "string",
            },
            {
                "name": "subcategory", 
                "description": "The subcategory of the content (e.g., 'General Info', 'Tech Fest (SDIT Spark)', 'About SDIT')",
                "type": "string",
            },
            {
                "name": "source",
                "description": "The source of the content (e.g., 'generated_template', 'official')",
                "type": "string",
            },
            {
                "name": "title",
                "description": "The title of the content entry",
                "type": "string",
            },
            {
                "name": "content_type",
                "description": "Type of content: 'templated' for placeholder information or 'factual' for real information",
                "type": "string",
            },
            {
                "name": "has_contact_info",
                "description": "Whether the content contains contact information (phone numbers, emails)",
                "type": "boolean",
            },
            {
                "name": "has_location_info",
                "description": "Whether the content contains location or address information",
                "type": "boolean",
            },
            {
                "name": "has_course_info",
                "description": "Whether the content contains information about courses or programs",
                "type": "boolean",
            },
        ]
        
        document_content_description = """
        Information about Shree Devi College including:
        - Campus details (locations, facilities, infrastructure)
        - Academic programs (courses, departments, admissions)
        - Student services (FAQs, procedures, support)
        - Events and activities (cultural, technical, sports)
        - General college information (history, mission, contact details)
        """
        
        try:
            # Create AttributeInfo objects if available
            if AttributeInfo is not None:
                attr_infos = [AttributeInfo(**field) for field in metadata_field_info]
            else:
                # Fallback to dictionary format
                attr_infos = metadata_field_info
            
            retriever = SelfQueryRetriever.from_llm(
                self.llm,
                vectorstore,
                document_content_description,
                attr_infos,
                enable_limit=True,  # Allow specifying number of results
                use_original_query=False,  # Use constructed query for better results
                verbose=False  # Set to True for debugging
            )
            
            print("✅ Self-query retriever created successfully")
            return retriever
            
        except Exception as e:
            print(f"❌ Failed to create self-query retriever: {e}")
            return None
    
    def create_enhanced_retriever(self, vectorstore: FAISS) -> Callable:
        """Create an enhanced retriever with manual query processing"""
        
        def enhanced_search(query: str, k: int = 5) -> List[Document]:
            """Enhanced search with manual query processing"""
            
            # Process query for better retrieval
            processed_query, detected_categories = self.enhanced_query_processing(query)
            
            # Get base results
            results = vectorstore.similarity_search(processed_query, k=k*2)  # Get more results initially
            
            # Filter and rerank based on detected categories
            if detected_categories:
                filtered_results = []
                for result in results:
                    if result.metadata.get('category') in detected_categories:
                        filtered_results.append(result)
                
                # If we have filtered results, use them, otherwise use original results
                if filtered_results:
                    results = filtered_results[:k]
                else:
                    results = results[:k]
            else:
                results = results[:k]
            
            # Boost results based on metadata
            boosted_results = []
            for result in results:
                boost = self.calculate_confidence_score(result, query)
                boosted_results.append((result, boost))
            
            # Sort by boost and return top k
            boosted_results.sort(key=lambda x: x[1], reverse=True)
            return [result for result, boost in boosted_results[:k]]
        
        return enhanced_search
    
    def build_optimized_index(self, dataset_file: str = DATASET_FILE):
        """Build optimized FAISS index with enhanced processing"""
        
        print("🔄 Building optimized RAG index...")
        
        # Initialize local LLM if not already done
        if self.llm is None:
            self.initialize_local_llm()
        
        # Load and validate data
        valid_data = self.load_and_validate_data(dataset_file)
        
        if not valid_data:
            print("❌ No valid data found")
            return
        
        # Create enhanced documents
        docs = self.create_enhanced_documents(valid_data)
        
        # Enhanced chunking
        chunker = self.create_semantic_chunker()
        chunks = chunker.split_documents(docs)
        
        print(f"✅ Created {len(chunks)} chunks from documents")
        
        # Further enhance chunks
        enhanced_chunks = self.enhance_document_processing(chunks)
        
        print(f"✅ Enhanced {len(enhanced_chunks)} chunks")
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(enhanced_chunks, self.embeddings)
        
        # Try to create self-query retriever
        if SELF_QUERY_AVAILABLE:
            self.retriever = self.create_self_query_retriever(self.vectorstore)
        
        # If self-query fails, create enhanced retriever
        if self.retriever is None:
            print("🔄 Creating enhanced retriever with manual processing")
            self.retriever = self.create_enhanced_retriever(self.vectorstore)
        
        # Ensure index directory exists
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        
        # Save index
        self.vectorstore.save_local(INDEX_PATH)
        print("✅ Optimized RAG index built and saved!")
    
    def load_optimized_index(self) -> bool:
        """Load index with memory mapping for better performance"""
        
        if not os.path.exists(INDEX_PATH):
            print("⚠️ Optimized index not found, building new index...")
            self.build_optimized_index()
            return True
        
        try:
            self.vectorstore = FAISS.load_local(
                INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            
            # Initialize local LLM if not already done
            if self.llm is None:
                self.initialize_local_llm()
            
            # Try to create self-query retriever
            if SELF_QUERY_AVAILABLE:
                self.retriever = self.create_self_query_retriever(self.vectorstore)
            
            # If self-query fails, create enhanced retriever
            if self.retriever is None:
                print("🔄 Creating enhanced retriever with manual processing")
                self.retriever = self.create_enhanced_retriever(self.vectorstore)
            
            print("✅ Optimized RAG index loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading optimized index: {e}")
            print("🔄 Building new index...")
            self.build_optimized_index()
            return True
    
    def enhanced_query_processing(self, query: str) -> tuple:
        """Process queries to improve retrieval accuracy"""
        
        # Add query prefixes for BGE model
        if not query.startswith("Represent this sentence for retrieval:"):
            processed_query = f"Represent this sentence for retrieval: {query}"
        else:
            processed_query = query
        
        # Extract potential metadata filters
        category_keywords = {
            "course": "Departments & Courses",
            "admission": "Admissions & Academics", 
            "facility": "Campus Facilities",
            "event": "Events & Activities",
            "placement": "Placement & Training",
            "college": "College Information",
            "campus": "College Information",
            "department": "Departments & Courses",
            "program": "Departments & Courses",
            "degree": "Departments & Courses",
            "hostel": "Campus Facilities",
            "library": "Campus Facilities",
            "lab": "Campus Facilities",
            "sport": "Events & Activities",
            "cultural": "Events & Activities",
            "technical": "Events & Activities",
            "fee": "Admissions & Academics",
            "scholarship": "Admissions & Academics",
            "contact": "College Information",
            "address": "College Information",
            "location": "College Information"
        }
        
        detected_categories = []
        for keyword, category in category_keywords.items():
            if keyword.lower() in query.lower():
                detected_categories.append(category)
        
        return processed_query, list(set(detected_categories))
    
    @lru_cache(maxsize=1000)
    def cached_search(self, query_hash: str, top_k: int = 5) -> str:
        """Cache search results for common queries"""
        
        # Get original query from hash (simplified - in production, use proper hashing)
        original_query = query_hash.replace("Represent this sentence for retrieval: ", "")
        
        # Perform search
        return self.optimized_search(original_query, top_k)
    
    def optimized_search(self, query: str, top_k: int = 5) -> str:
        """Optimized search with enhanced capabilities"""
        
        # Load index if not loaded
        if self.vectorstore is None:
            if not self.load_optimized_index():
                return "Error: Could not load or build RAG index"
        
        # Process query
        processed_query, detected_categories = self.enhanced_query_processing(query)
        
        try:
            # Try self-query or enhanced retriever
            if self.retriever:
                if hasattr(self.retriever, 'invoke'):
                    # SelfQueryRetriever
                    results = self.retriever.invoke(query)
                else:
                    # Enhanced retriever (manual processing)
                    results = self.retriever(query, k=top_k)
            else:
                # Fallback to basic search
                results = self.vectorstore.similarity_search(processed_query, k=top_k)
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            # Fallback to basic search
            try:
                results = self.vectorstore.similarity_search(processed_query, k=top_k)
            except Exception as fallback_error:
                print(f"❌ Fallback search error: {fallback_error}")
                return "Sorry, an error occurred while searching for information."
        
        return self.format_results(results, query)
    
    def format_results(self, results: List[Document], original_query: str) -> str:
        """Enhanced result formatting with confidence scoring"""
        
        if not results:
            return "Shishishi! Looks like there's no information in Shree Devi College's database. Go ahead and request a callback from the college to find out more!"
        
        formatted_results = []
        
        # Check if query is about courses/programs
        is_course_query = any(keyword in original_query.lower() for keyword in [
            "course", "program", "department", "study", "subject", "degree", "diploma", "certification"
        ])
        
        for i, result in enumerate(results, 1):
            meta = result.metadata
            content = result.page_content
            
            # Extract clean content
            clean_content = self.extract_clean_content(content)
            
            # Calculate confidence score
            confidence_score = self.calculate_confidence_score(result, original_query)
            
            # Build formatted result
            formatted_result = f"""
Result {i}:
Source: {meta['title']} (Category: {meta['category']})
Confidence: {confidence_score:.2f}
Content: {clean_content}
"""
            
            # Add warnings for templated content
            if meta.get('content_type') == 'templated':
                formatted_result += "\n⚠️ NOTE: This is templated information and may not represent actual details. "
                formatted_result += "Please verify with the college directly for accurate information."
            
            formatted_results.append(formatted_result)
        
        # Join all results
        final_response = "\n---\n".join(formatted_results)
        
        # Add special note for course queries with templated content
        if is_course_query:
            all_templated = all(
                r.metadata.get('content_type') == 'templated' for r in results
            )
            
            if all_templated:
                final_response += f"""

⚠️ IMPORTANT: The information above is templated and does not contain specific course details. 
Shree Devi College offers various undergraduate and postgraduate programs, but the specific course details are not available in my current database. 
I recommend contacting the college directly at +91 (824) 2254104 or visiting their campus for accurate course information."""
        
        return final_response
    
    def extract_clean_content(self, content: str) -> str:
        """Extract clean content from structured format"""
        
        # Remove metadata markers
        content = re.sub(r'\[CATEGORY: [^\]]+\]', '', content)
        content = re.sub(r'\[SUBCATEGORY: [^\]]+\]', '', content)
        content = re.sub(r'\[TITLE: [^\]]+\]', '', content)
        content = re.sub(r'\[CONTENT: ([^\]]+)\]', r'\1', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n', content)
        content = content.strip()
        
        return content
    
    def calculate_confidence_score(self, result: Document, query: str) -> float:
        """Calculate confidence score for result"""
        
        # Base score
        score = 0.5
        
        meta = result.metadata
        
        # Boost score for factual content
        if meta.get('content_type') == 'factual':
            score += 0.2
        
        # Boost score for relevant category matches
        query_lower = query.lower()
        
        category_boosts = {
            'course': meta.get('has_course_info'),
            'admission': meta.get('has_admission_info'),
            'contact': meta.get('has_contact_info'),
            'location': meta.get('has_location_info'),
            'event': meta.get('has_event_info'),
            'facility': any(keyword in query_lower for keyword in ['lab', 'library', 'hostel', 'facility']),
        }
        
        for keyword, has_info in category_boosts.items():
            if keyword in query_lower and has_info:
                score += 0.15
        
        # Boost score for content length (longer content might be more comprehensive)
        content_length = meta.get('content_length', 0)
        if content_length > 200:
            score += 0.05
        
        return min(score, 1.0)
    
    def evaluate_system(self, test_queries: List[str]) -> List[Dict[str, Any]]:
        """Evaluate the RAG system performance"""
        
        results = []
        
        for query in test_queries:
            start_time = time.time()
            search_results = self.optimized_search(query, top_k=3)
            end_time = time.time()
            
            # Basic evaluation metrics
            evaluation = {
                'query': query,
                'response_time': end_time - start_time,
                'has_results': len(search_results) > 0,
                'result_count': len(search_results.split('---')) if '---' in search_results else 1,
                'has_templated_warning': '⚠️ NOTE' in search_results,
                'confidence_score': self.extract_average_confidence(search_results)
            }
            
            results.append(evaluation)
        
        return results
    
    def extract_average_confidence(self, search_results: str) -> float:
        """Extract average confidence score from search results"""
        
        confidence_matches = re.findall(r'Confidence: (\d+\.\d+)', search_results)
        if confidence_matches:
            return sum(float(score) for score in confidence_matches) / len(confidence_matches)
        return 0.0

# Global instance
rag_system = OptimizedRAGSystem()

# Legacy functions for backward compatibility
def build_rag_index():
    """Legacy function - builds optimized index"""
    rag_system.build_optimized_index()

def load_rag_index():
    """Legacy function - loads optimized index"""
    return rag_system.load_optimized_index()

def rag_lookup(query: str, top_k: int = 5) -> str:
    """Legacy function - performs optimized search"""
    return rag_system.optimized_search(query, top_k)

def validate_dataset_file():
    """Legacy function - validates dataset file"""
    valid_data = rag_system.load_and_validate_data(DATASET_FILE)
    print(f"✅ Validation complete: {len(valid_data)} valid entries")
    return len(valid_data) > 0

if __name__ == "__main__":
    # Run validation and build index
    validate_dataset_file()
    build_rag_index()