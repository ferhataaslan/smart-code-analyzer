#!/usr/bin/env python3
import unittest
import json
import os
import time
import sqlite3
import threading
from src.core.automator import AutomatedCollector

class TestAutomator(unittest.TestCase):
    def setUp(self):
        self.state_file = "test_data/states/automation_state.json"
        
        # Backup main DB files if they exist to prevent tests from modifying them
        self.db_backups = {}
        for db in ("data/dbs/vector_store.db", "data/dbs/document_store.db"):
            if os.path.exists(db):
                self.db_backups[db] = db + ".bak"
                if os.path.exists(db + ".bak"):
                    os.remove(db + ".bak")
                os.rename(db, db + ".bak")

        # Create fresh mock databases
        conn_vec = sqlite3.connect("data/dbs/vector_store.db")
        conn_vec.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                entity_id TEXT PRIMARY KEY,
                source_system TEXT,
                normalized_structure TEXT,
                applied_algorithm TEXT,
                ast_metadata TEXT,
                nl_alignment TEXT,
                security_context TEXT,
                data_flow_graph TEXT,
                complexity TEXT,
                code_hash TEXT,
                created_at TEXT
            )
        """)
        conn_vec.close()

        conn_doc = sqlite3.connect("data/dbs/document_store.db")
        conn_doc.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                entity_id TEXT PRIMARY KEY,
                source_system TEXT,
                raw_snippet TEXT,
                doc_metadata TEXT,
                code_hash TEXT,
                created_at TEXT
            )
        """)
        conn_doc.close()

    def tearDown(self):
        # Remove state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

        # Remove mock DB files
        for db in ("data/dbs/vector_store.db", "data/dbs/document_store.db"):
            if os.path.exists(db):
                os.remove(db)

        # Restore backups
        for original, backup in self.db_backups.items():
            if os.path.exists(backup):
                os.rename(backup, original)

    def test_state_load_save(self):
        automator = AutomatedCollector(state_file=self.state_file)
        state = automator.load_state()
        self.assertEqual(state["status"], "RUNNING")
        
        # Update and save
        state["status"] = "PAUSED"
        state["github_page"] = 42
        automator.save_state(state)
        
        # Load again to check persistence
        new_state = automator.load_state()
        self.assertEqual(new_state["status"], "PAUSED")
        self.assertEqual(new_state["github_page"], 42)

    def test_pause_resume_mechanism(self):
        automator = AutomatedCollector(state_file=self.state_file)
        
        # Set to PAUSED
        state = automator.load_state()
        state["status"] = "PAUSED"
        automator.save_state(state)

        paused_done = False
        def run_pause_check():
            nonlocal paused_done
            automator._check_pause()
            paused_done = True

        # Run check_pause in a background thread
        t = threading.Thread(target=run_pause_check)
        t.start()
        
        # Let it block for a moment
        time.sleep(0.5)
        self.assertFalse(paused_done) # Still blocked

        # Change status to RUNNING
        state["status"] = "RUNNING"
        automator.save_state(state)
        
        # Wait for the thread to resume and complete
        t.join(timeout=2)
        self.assertTrue(paused_done) # Successfully resumed and finished

    def test_capacity_flush_reclaim(self):
        # Set BATCH_THRESHOLD to 5 for testing capacity cleanup
        os.environ["BATCH_THRESHOLD"] = "5"
        
        automator = AutomatedCollector(state_file=self.state_file)
        
        # Populate databases with 5 dummy records
        conn_vec = sqlite3.connect("data/dbs/vector_store.db")
        conn_doc = sqlite3.connect("data/dbs/document_store.db")
        
        for i in range(5):
            conn_vec.execute(
                "INSERT INTO vectors (entity_id, source_system, normalized_structure, applied_algorithm) VALUES (?, ?, ?, ?)",
                (f"id-{i}", "TEST", f"norm-code-{i}", "test-algo")
            )
            conn_doc.execute(
                "INSERT INTO documents (entity_id, source_system, raw_snippet, code_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (f"id-{i}", "TEST", f"raw-code-{i}", f"hash-{i}", "2026-06-02")
            )
        conn_vec.commit()
        conn_doc.commit()
        
        # Run capacity check
        automator._check_capacity()
        
        # Verify document store raw_snippets are cleared (set to '') but rows and hashes are kept
        cursor_doc = conn_doc.cursor()
        cursor_doc.execute("SELECT raw_snippet, code_hash FROM documents")
        doc_rows = cursor_doc.fetchall()
        self.assertEqual(len(doc_rows), 5)
        for row in doc_rows:
            self.assertEqual(row[0], "") # raw_snippet is cleared
            self.assertTrue(row[1].startswith("hash-")) # code_hash kept

        # Verify vectors table is completely empty (deleted) so they won't re-upload
        cursor_vec = conn_vec.cursor()
        cursor_vec.execute("SELECT COUNT(*) FROM vectors")
        count = cursor_vec.fetchone()[0]
        self.assertEqual(count, 0)

        conn_vec.close()
        conn_doc.close()

if __name__ == "__main__":
    unittest.main()
