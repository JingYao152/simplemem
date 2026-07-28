"""
SimpleMem - Efficient Lifelong Memory for LLM Agents
Main system class integrating all components
"""
from typing import List, Optional
from simplemem.core.models.memory_entry import Dialogue, MemoryEntry
from simplemem.core.utils.llm_client import LLMClient
from simplemem.core.utils.embedding import EmbeddingModel
from simplemem.core.database.vector_store import VectorStore
from simplemem.core.memory_builder import MemoryBuilder
from simplemem.core.memweaver import MemWeaver
from simplemem.core.memweaver.dual_view import state_anchor_table_name
from simplemem.core.hybrid_retriever import HybridRetriever
from simplemem.core.answer_generator import AnswerGenerator
from simplemem.core.settings import settings as config


class SimpleMemSystem:
    """
    SimpleMem Main System

    Three-stage pipeline:
    1. Semantic Structured Compression (Section 3.1): add_dialogue() -> MemoryBuilder -> VectorStore
    2. Online Semantic Synthesis (Section 3.2): Intra-session consolidation during write
    3. Intent-Aware Retrieval Planning (Section 3.3): ask() -> HybridRetriever -> AnswerGenerator

    With MemWeaver enabled (docs/memweaver-design.md), stages 1-2 are replaced by
    the session-atomic Call A/B fabric pipeline and the retrieval base gains
    as-of validity filtering; everything else stays SimpleMem-native so the two
    arms stay comparable.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        clear_db: bool = False,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
        enable_planning: Optional[bool] = None,
        enable_reflection: Optional[bool] = None,
        max_reflection_rounds: Optional[int] = None,
        enable_parallel_processing: Optional[bool] = None,
        max_parallel_workers: Optional[int] = None,
        enable_parallel_retrieval: Optional[bool] = None,
        max_retrieval_workers: Optional[int] = None,
        enable_memweaver: Optional[bool] = None,
        enable_weaving: Optional[bool] = None,
        enable_sweep: Optional[bool] = None,
        enable_recontext: Optional[bool] = None,
        enable_entity_profiles: Optional[bool] = None,
        enable_dual_view_state_anchors: Optional[bool] = None,
        enable_expand_rerank: Optional[bool] = None,
        enable_expansion: Optional[bool] = None,
        enable_rerank: Optional[bool] = None
    ):
        """
        Initialize system

        Args:
        - api_key: OpenAI API key
        - model: LLM model name
        - base_url: Custom OpenAI base URL (for compatible APIs)
        - db_path: Database path
        - table_name: Memory table name (for parallel processing)
        - clear_db: Whether to clear existing database
        - enable_thinking: Enable deep thinking mode (for Qwen and compatible models)
        - use_streaming: Enable streaming responses
        - enable_planning: Enable multi-query planning for retrieval (None=use config default)
        - enable_reflection: Enable reflection-based additional retrieval (None=use config default)
        - max_reflection_rounds: Maximum number of reflection rounds (None=use config default)
        - enable_parallel_processing: Enable parallel processing for memory building (None=use config default)
        - max_parallel_workers: Maximum number of parallel workers for memory building (None=use config default)
        - enable_parallel_retrieval: Enable parallel processing for retrieval queries (None=use config default)
        - max_retrieval_workers: Maximum number of parallel workers for retrieval (None=use config default)
        - enable_memweaver: Use the MemWeaver fabric write pipeline + as-of retrieval (None=use config default)
        - enable_weaving: Enable typed weave operations (ablation switch, None=use config default)
        - enable_sweep: Enable the finalize cross-thread sweep (ablation switch, None=use config default)
        - enable_recontext: Enable context-inheriting embeddings + re-embedding (ablation switch, None=use config default)
        - enable_entity_profiles: Enable entity profiles in the pool (ablation switch, None=use config default)
        - enable_expand_rerank: Enable the whole P2 read stage (compound ablation switch, None=use config default)
        - enable_expansion: Enable one-hop fabric expansion + chain annotation (C3 ablation switch, None=use config default)
        - enable_rerank: Enable the cross-encoder reranker (inherited component; keep on for both A/B arms)
        """
        print("=" * 60)
        print("Initializing SimpleMem System")
        print("=" * 60)

        # Initialize core components
        self.llm_client = LLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming
        )
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(
            db_path=db_path,
            embedding_model=self.embedding_model,
            table_name=table_name
        )
        self.enable_dual_view_state_anchors = (
            enable_dual_view_state_anchors
            if enable_dual_view_state_anchors is not None
            else getattr(config, 'ENABLE_DUAL_VIEW_STATE_ANCHORS', False)
        )
        primary_table_name = table_name or config.MEMORY_TABLE_NAME
        self.state_anchor_store = (
            VectorStore(
                db_path=db_path,
                embedding_model=self.embedding_model,
                table_name=state_anchor_table_name(primary_table_name),
            )
            if self.enable_dual_view_state_anchors
            else None
        )

        if clear_db:
            print("\nClearing existing database...")
            self.vector_store.clear()
            if self.state_anchor_store is not None:
                self.state_anchor_store.clear()

        # Initialize three major modules
        self.memory_builder = MemoryBuilder(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            enable_parallel_processing=enable_parallel_processing,
            max_parallel_workers=max_parallel_workers
        )

        self.enable_memweaver = (
            enable_memweaver
            if enable_memweaver is not None
            else getattr(config, 'ENABLE_MEMWEAVER', False)
        )
        self.memweaver: Optional[MemWeaver] = None
        if self.enable_memweaver:
            self.memweaver = MemWeaver(
                llm_client=self.llm_client,
                vector_store=self.vector_store,
                enable_weaving=enable_weaving,
                enable_sweep=enable_sweep,
                max_parallel_workers=max_parallel_workers,
                fallback_extractor=self.memory_builder,
                enable_recontext=enable_recontext,
                enable_entity_profiles=enable_entity_profiles,
                state_anchor_store=self.state_anchor_store,
                enable_dual_view_state_anchors=self.enable_dual_view_state_anchors,
            )
            print(
                "\nMemWeaver write pipeline enabled "
                f"(weaving={self.memweaver.enable_weaving}, "
                f"sweep={self.memweaver.enable_sweep}, "
                f"recontext={self.memweaver.enable_recontext}, "
                f"dual_view_state_anchors={self.memweaver.enable_dual_view_state_anchors}, "
                f"profiles={self.memweaver.enable_entity_profiles}, "
                f"temperature={self.memweaver.temperature})"
            )

        # The component the write path routes to (MemoryBuilder = pure SimpleMem)
        self.writer = self.memweaver or self.memory_builder

        self.hybrid_retriever = HybridRetriever(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_reflection_rounds=max_reflection_rounds,
            enable_parallel_retrieval=enable_parallel_retrieval,
            max_retrieval_workers=max_retrieval_workers,
            enable_memweaver=self.enable_memweaver,
            enable_expand_rerank=enable_expand_rerank,
            enable_expansion=enable_expansion,
            enable_rerank=enable_rerank,
            state_anchor_store=self.state_anchor_store,
            enable_dual_view_state_anchors=self.enable_dual_view_state_anchors,
        )

        # The supersede-chain annotation is what makes expansion-recovered history
        # readable, so it follows the expansion switch (C3) rather than the
        # inherited reranker.
        self.answer_generator = AnswerGenerator(
            llm_client=self.llm_client,
            annotate_chains=self.hybrid_retriever.enable_expansion
        )
        if self.hybrid_retriever.enable_expand_rerank:
            print(
                "P2 read stage enabled "
                f"(expansion={self.hybrid_retriever.enable_expansion}, "
                f"rerank={self.hybrid_retriever.enable_rerank}, "
                f"model={self.hybrid_retriever.reranker.model_name}, "
                f"top_k={self.hybrid_retriever.reranker.top_k})"
            )

        print("\nSystem initialization complete!")
        print("=" * 60)

    def add_dialogue(self, speaker: str, content: str, timestamp: Optional[str] = None):
        """
        Add a single dialogue

        Args:
        - speaker: Speaker name
        - content: Dialogue content
        - timestamp: Timestamp (ISO 8601 format)
        """
        dialogue_id = self.writer.processed_count + len(self.writer.dialogue_buffer) + 1
        dialogue = Dialogue(
            dialogue_id=dialogue_id,
            speaker=speaker,
            content=content,
            timestamp=timestamp
        )
        self.writer.add_dialogue(dialogue)

    def add_dialogues(self, dialogues: List[Dialogue]):
        """
        Batch add dialogues

        Args:
        - dialogues: List of dialogues
        """
        self.writer.add_dialogues(dialogues)

    def finalize(self):
        """
        Finalize dialogue input, process any remaining buffer (safety check)
        Note: In parallel mode, remaining dialogues are already processed

        With MemWeaver enabled this also runs the cross-thread supersede sweep
        and optimizes the store.
        """
        if self.memweaver is not None:
            self.memweaver.finalize()
        else:
            self.memory_builder.process_remaining()

    def ask(self, question: str) -> str:
        """
        Ask question - Core Q&A interface

        Args:
        - question: User question

        Returns:
        - Answer
        """
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        # Stage 3: Intent-Aware Retrieval Planning
        contexts = self.hybrid_retriever.retrieve(question)

        # Generate answer from retrieved context C_q
        answer = self.answer_generator.generate_answer(question, contexts)

        print("\nAnswer:")
        print(answer)
        print("=" * 60 + "\n")

        return answer

    def get_all_memories(self) -> List[MemoryEntry]:
        """
        Get all memory entries (for debugging)
        """
        return self.vector_store.get_all_entries()

    def print_memories(self):
        """
        Print all memory entries (for debugging)
        """
        memories = self.get_all_memories()
        print("\n" + "=" * 60)
        print(f"All Memory Entries ({len(memories)} total)")
        print("=" * 60)

        for i, memory in enumerate(memories, 1):
            print(f"\n[Entry {i}]")
            print(f"ID: {memory.entry_id}")
            print(f"Restatement: {memory.lossless_restatement}")
            if memory.timestamp:
                print(f"Time: {memory.timestamp}")
            if memory.location:
                print(f"Location: {memory.location}")
            if memory.persons:
                print(f"Persons: {', '.join(memory.persons)}")
            if memory.entities:
                print(f"Entities: {', '.join(memory.entities)}")
            if memory.topic:
                print(f"Topic: {memory.topic}")
            print(f"Keywords: {', '.join(memory.keywords)}")

        print("\n" + "=" * 60)


# Convenience function
def create_system(
    clear_db: bool = False,
    enable_planning: Optional[bool] = None,
    enable_reflection: Optional[bool] = None,
    max_reflection_rounds: Optional[int] = None,
    enable_parallel_processing: Optional[bool] = None,
    max_parallel_workers: Optional[int] = None,
    enable_parallel_retrieval: Optional[bool] = None,
    max_retrieval_workers: Optional[int] = None,
    enable_memweaver: Optional[bool] = None,
    enable_weaving: Optional[bool] = None,
    enable_sweep: Optional[bool] = None,
    enable_recontext: Optional[bool] = None,
    enable_entity_profiles: Optional[bool] = None,
    enable_dual_view_state_anchors: Optional[bool] = None,
    enable_expand_rerank: Optional[bool] = None,
    enable_expansion: Optional[bool] = None,
    enable_rerank: Optional[bool] = None
) -> SimpleMemSystem:
    """
    Create SimpleMem system instance (uses config.py defaults when None)
    """
    return SimpleMemSystem(
        clear_db=clear_db,
        enable_planning=enable_planning,
        enable_reflection=enable_reflection,
        max_reflection_rounds=max_reflection_rounds,
        enable_parallel_processing=enable_parallel_processing,
        max_parallel_workers=max_parallel_workers,
        enable_parallel_retrieval=enable_parallel_retrieval,
        max_retrieval_workers=max_retrieval_workers,
        enable_memweaver=enable_memweaver,
        enable_weaving=enable_weaving,
        enable_sweep=enable_sweep,
        enable_recontext=enable_recontext,
        enable_entity_profiles=enable_entity_profiles,
        enable_dual_view_state_anchors=enable_dual_view_state_anchors,
        enable_expand_rerank=enable_expand_rerank,
        enable_expansion=enable_expansion,
        enable_rerank=enable_rerank
    )


if __name__ == "__main__":
    # Quick test with Qwen3 integration
    print("🚀 Running SimpleMem Quick Test with Qwen3...")

    system = create_system(clear_db=True)
    print(f"📌 Using embedding model: {system.memory_builder.vector_store.embedding_model.model_name}")
    print(f"📌 Model type: {system.memory_builder.vector_store.embedding_model.model_type}")

    # Add some test dialogues
    system.add_dialogue("Alice", "Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product", "2025-11-15T14:30:00")
    system.add_dialogue("Bob", "Okay, I'll prepare the materials", "2025-11-15T14:31:00")
    system.add_dialogue("Alice", "Remember to bring the market research report from last time", "2025-11-15T14:32:00")

    # Finalize input
    system.finalize()

    # View memories
    system.print_memories()

    # Ask questions (with new features)
    print("\n🔍 Testing retrieval with planning and reflection...")
    system.ask("When will Alice and Bob meet?")
    
    print("\n🔍 Testing adversarial question (reflection disabled)...")
    question = "What is Alice's favorite food?"
    contexts = system.hybrid_retriever.retrieve(question, enable_reflection=False)
    answer = system.answer_generator.generate_answer(question, contexts)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    print("\n✅ Quick test completed!")
    print("\n💡 To run comprehensive tests: python test_qwen3_integration.py")
