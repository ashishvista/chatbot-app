#!/usr/bin/env python3
"""
Performance Profiling for RAG Pipeline

This script helps you identify performance bottlenecks in your RAG chatbot.
"""

import cProfile
import pstats
import time
import memory_profiler
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def profile_rag_pipeline():
    """Profile the RAG pipeline performance"""
    
    print("🔍 Profiling RAG Pipeline Performance...")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    try:
        # Start profiling
        profiler.enable()
        
        # Import and run pipeline
        from chatbot.rag_pipeline import RAGPipeline
        
        # Initialize pipeline
        start_time = time.time()
        pipeline = RAGPipeline()
        init_time = time.time() - start_time
        
        # Test queries
        test_queries = [
            "What are baby milestones at 6 months?",
            "How to handle fever in babies?",
            "Signs of developmental delays?",
            "Baby nutrition guidelines?",
            "When to call pediatrician?"
        ]
        
        query_times = []
        for query in test_queries:
            start_time = time.time()
            response = pipeline.generate_response(query)
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        # Stop profiling
        profiler.disable()
        
        # Print results
        print(f"\n⏱️ Performance Results:")
        print(f"Pipeline initialization: {init_time:.2f} seconds")
        print(f"Average query time: {sum(query_times)/len(query_times):.2f} seconds")
        print(f"Min query time: {min(query_times):.2f} seconds")
        print(f"Max query time: {max(query_times):.2f} seconds")
        
        # Save profile stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Print top time-consuming functions
        print(f"\n🔥 Top Time-Consuming Functions:")
        stats.print_stats(20)
        
        # Save to file
        stats.dump_stats('rag_profile.prof')
        print(f"\n💾 Profile saved to rag_profile.prof")
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()

@memory_profiler.profile
def profile_memory_usage():
    """Profile memory usage during pipeline operations"""
    
    print("🧠 Profiling Memory Usage...")
    
    # Import after profiler setup
    from chatbot.rag_pipeline import RAGPipeline
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Test query
    response = pipeline.generate_response("What are baby milestones?")
    
    return response

def analyze_bottlenecks():
    """Analyze potential bottlenecks in the pipeline"""
    
    print("🔍 Analyzing Performance Bottlenecks...")
    
    import psutil
    import threading
    
    # Monitor system resources
    def monitor_resources():
        """Monitor CPU and memory usage"""
        process = psutil.Process()
        
        print("\n📊 System Resources During Execution:")
        print(f"CPU Count: {psutil.cpu_count()}")
        print(f"Available Memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
        
        # Monitor for 30 seconds
        for i in range(30):
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if i % 5 == 0:  # Print every 5 seconds
                print(f"Time {i}s: CPU {cpu_percent:.1f}%, Memory {memory_mb:.1f} MB")
            
            time.sleep(1)
    
    # Start monitoring in background
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run pipeline operations
    try:
        from chatbot.rag_pipeline import RAGPipeline
        
        print("🚀 Starting pipeline operations...")
        pipeline = RAGPipeline()
        
        # Time individual components
        timings = {}
        
        # Test document retrieval
        start_time = time.time()
        docs = pipeline.get_similar_documents("baby milestones", k=5)
        timings['document_retrieval'] = time.time() - start_time
        
        # Test response generation
        start_time = time.time()
        response = pipeline.generate_response("What are normal baby milestones?")
        timings['response_generation'] = time.time() - start_time
        
        print(f"\n⏱️ Component Timings:")
        for component, timing in timings.items():
            print(f"{component}: {timing:.3f} seconds")
        
        # Wait for monitoring to complete
        time.sleep(5)
        
    except Exception as e:
        print(f"❌ Bottleneck analysis failed: {e}")

def generate_performance_report():
    """Generate a comprehensive performance report"""
    
    print("📈 Generating Performance Report...")
    
    report = []
    report.append("# RAG Pipeline Performance Report")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    try:
        # System info
        import psutil
        report.append("## System Information")
        report.append(f"- CPU Cores: {psutil.cpu_count()}")
        report.append(f"- Total Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
        report.append(f"- Available Memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
        report.append("")
        
        # Run performance tests
        from chatbot.rag_pipeline import RAGPipeline
        
        # Initialization timing
        start_time = time.time()
        pipeline = RAGPipeline()
        init_time = time.time() - start_time
        
        report.append("## Performance Metrics")
        report.append(f"- Pipeline Initialization: {init_time:.2f} seconds")
        
        # Query performance
        test_queries = [
            "What are baby milestones?",
            "How to handle fever?",
            "Signs of developmental delays?",
            "Baby nutrition guidelines?",
            "Emergency situations?"
        ]
        
        query_times = []
        for query in test_queries:
            start_time = time.time()
            response = pipeline.generate_response(query)
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        avg_time = sum(query_times) / len(query_times)
        report.append(f"- Average Query Time: {avg_time:.2f} seconds")
        report.append(f"- Fastest Query: {min(query_times):.2f} seconds")
        report.append(f"- Slowest Query: {max(query_times):.2f} seconds")
        report.append("")
        
        # Recommendations
        report.append("## Performance Recommendations")
        
        if init_time > 30:
            report.append("- ⚠️ Slow initialization - Consider caching models")
        
        if avg_time > 10:
            report.append("- ⚠️ Slow query response - Consider optimizing retrieval")
        
        if avg_time < 3:
            report.append("- ✅ Good query response time")
        
        if init_time < 15:
            report.append("- ✅ Fast initialization")
        
        # Save report
        with open("performance_report.md", "w") as f:
            f.write("\n".join(report))
        
        print("✅ Performance report saved to performance_report.md")
        
        # Print summary
        print(f"\n📊 Performance Summary:")
        print(f"Initialization: {init_time:.2f}s")
        print(f"Average Query: {avg_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")

if __name__ == "__main__":
    print("🔧 RAG Pipeline Performance Analyzer")
    print("=" * 50)
    
    # Choose what to run
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline Performance Profiler")
    parser.add_argument("--profile", action="store_true", help="Run cProfile profiling")
    parser.add_argument("--memory", action="store_true", help="Run memory profiling")
    parser.add_argument("--bottlenecks", action="store_true", help="Analyze bottlenecks")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--all", action="store_true", help="Run all profiling")
    
    args = parser.parse_args()
    
    if args.all or not any([args.profile, args.memory, args.bottlenecks, args.report]):
        # Run everything if no specific option chosen
        profile_rag_pipeline()
        analyze_bottlenecks()
        generate_performance_report()
    else:
        if args.profile:
            profile_rag_pipeline()
        if args.memory:
            profile_memory_usage()
        if args.bottlenecks:
            analyze_bottlenecks()
        if args.report:
            generate_performance_report()
