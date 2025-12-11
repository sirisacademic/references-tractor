# test_installation.py
import sys
import json

def test_imports():
    """Test all critical imports work"""
    try:
        from references_tractor import ReferencesTractor
        print("✅ Main package import: SUCCESS")
        
        # Test submodules
        from references_tractor.search import search_api
        from references_tractor.search import citation_formatter
        print("✅ Submodule imports: SUCCESS")
        
        # Test key dependencies
        import torch
        import transformers
        import pandas
        import numpy
        print("✅ Key dependencies: SUCCESS")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic package functionality"""
    try:
        from references_tractor import ReferencesTractor
        ref_tractor = ReferencesTractor()
        print("✅ Package initialization: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_example():
    """Test package usage with real citation"""
    try:
        from references_tractor import ReferencesTractor
        print("\n🔍 Testing citation linking...")
        
        # Initialize the parser
        ref_tractor = ReferencesTractor()
        
        # Raw citation text
        citation = "Perrier J, Amato JN, Berthelon C, Bocca ML. Primary insomnia patients' performances during simulated car following and urban driving in the afternoon. J Sleep Res. 2019 Aug;28(4):e12847. DOI: 10.1111/jsr.12847."
        
        # Parse and link the citation
        #result = ref_tractor.link_citation(citation, api_target="openalex", output='simple')
        result = ref_tractor.link_citation_ensemble(citation)
        
        print("📝 Citation:", citation[:80] + "...")
        print("🔗 Result:", json.dumps(result, indent=2))
        print("✅ Example test: SUCCESS")
        
        return True
    except Exception as e:
        print(f"❌ Example test failed: {e}")
        return False

def test_pipeline_steps():
    """Test individual pipeline steps using existing methods"""
    try:
        from references_tractor import ReferencesTractor
        print("\n🔧 Testing pipeline steps...")
        
        ref_tractor = ReferencesTractor()
        
        # Step 1: Check if text is a citation (using prescreening pipeline)
        
        # Invalid citation
        citation = "This is not a citation"
        prescreening_result = ref_tractor.prescreening_pipeline(citation)[0]
        is_valid = prescreening_result["label"]
        print(f"Citation: {citation}")
        print(f"✅ Citation validation: {is_valid} (confidence: {prescreening_result.get('score', 'N/A'):.3f})")
 
        print()
 
        # Valid citation
        citation = "Perrier J, Amato JN, Berthelon C, Bocca ML. Primary insomnia patients' performances during simulated car following and urban driving in the afternoon. J Sleep Res. 2019 Aug;28(4):e12847. DOI: 10.1111/jsr.12847."
        
        prescreening_result = ref_tractor.prescreening_pipeline(citation)[0]       
        is_valid = prescreening_result["label"]
        print(f"Citation: {citation}")
        print(f"✅ Citation validation: {is_valid} (confidence: {prescreening_result.get('score', 'N/A'):.3f})")
 
        if is_valid:
            print()
            # Step 2: Extract entities using existing NER method
            entities = ref_tractor.process_ner_entities(citation)
            print(f"✅ Entity extraction: Found {len(entities)} entity types")
            print(f"   Entities: {list(entities.keys())}")
            if entities:
                # Show some example entities
                for entity_type, values in list(entities.items()):
                    print(f"   {entity_type}: {values}")
            
            print()
            # Step 3: Search for candidates using existing search method
            candidates = ref_tractor.search_api(entities, api="openalex")
            print(f"✅ Candidate search: Found {len(candidates)} candidates")
            
            if candidates:
                print()
                # Step 4: Generate formatted citations for ranking
                formatted_citations = [ref_tractor.generate_apa_citation(pub, api="openalex") for pub in candidates]
                pairwise_scores = [ref_tractor.select_pipeline(f"{citation} [SEP] {cit}") for cit in formatted_citations]
                print(f"✅ Candidate ranking: Scored {len(pairwise_scores)} candidates")
                
                # Show best match info
                if pairwise_scores:
                    best_idx = 0
                    best_score = pairwise_scores[0][0]
                    for i, score_result in enumerate(pairwise_scores):
                        if score_result[0].get('score', 0) > best_score.get('score', 0):
                            best_idx = i
                            best_score = score_result[0]
                    
                    print(f"   Best match score: {best_score.get('score', 'N/A'):.3f}")
                    print(f"   Best match: {formatted_citations[best_idx][:80]}...")
            
        print("✅ Pipeline steps test: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline steps test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_modernbert_support():
    """Test ModernBERT model loading"""
    try:
        from transformers import AutoModel, AutoTokenizer
        
        print("\n🤖 Testing ModernBERT support...")
        
        # Test loading ModernBERT
        model_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print("✅ ModernBERT support: SUCCESS")
        print(f"📝 Model: {model_name}")
        print(f"🔢 Parameters: ~149M")
        print(f"📏 Max length: 8192 tokens")
        
        return True
    except Exception as e:
        print(f"❌ ModernBERT test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing References Tractor installation...")
    print("=" * 50)
    
    all_ok = test_imports()
    if all_ok:
        print("\n✅ Import test passed!")
        all_ok = test_basic_functionality()
        
        if all_ok:
            print("✅ Basic functionality test passed!")
            all_ok = test_pipeline_steps()
            
            if all_ok:
                print("✅ Pipeline steps test passed!")
                all_ok = test_example()
                
                if all_ok:
                    print("✅ Example test passed!")
                    # Optional: Test ModernBERT (may take time to download)
                    print("\n" + "=" * 50)
                    print("Optional: Testing ModernBERT support (may download model)...")
                    modernbert_ok = test_modernbert_support()
                    if modernbert_ok:
                        print("✅ ModernBERT test passed!")
        
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ ALL CORE TESTS PASSED! Installation successful!")
    else:
        print("⚠️  SOME TESTS FAILED.")
