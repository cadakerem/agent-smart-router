import os
import time
import pytest
from llm_proxy_cli import CircuitBreaker

def test_circuit_breaker_isolation(tmp_path):
    # Test that different projects don't share state
    cb1 = CircuitBreaker("proj1", max_failures=2, cooldown_seconds=10)
    cb2 = CircuitBreaker("proj2", max_failures=2, cooldown_seconds=10)
    
    cb1.circuit_file = str(tmp_path / "cb1.json")
    cb1.lock_file = str(tmp_path / "cb1.json.lock")
    cb2.circuit_file = str(tmp_path / "cb2.json")
    cb2.lock_file = str(tmp_path / "cb2.json.lock")
    
    cb1.record_failure("modelA")
    cb1.record_failure("modelA")
    
    # cb1 should be tripped
    assert cb1.check_health("modelA") == False
    # cb2 should be unaffected
    assert cb2.check_health("modelA") == True

def test_circuit_breaker_cooldown(tmp_path):
    cb = CircuitBreaker("test", max_failures=1, cooldown_seconds=1)
    cb.circuit_file = str(tmp_path / "cb.json")
    cb.lock_file = str(tmp_path / "cb.json.lock")
    
    assert cb.check_health("modelB") == True
    cb.record_failure("modelB")
    
    assert cb.check_health("modelB") == False
    time.sleep(1.1)
    # After cooldown, should be healthy again
    assert cb.check_health("modelB") == True
    
def test_circuit_breaker_success_reset(tmp_path):
    cb = CircuitBreaker("test2", max_failures=2, cooldown_seconds=10)
    cb.circuit_file = str(tmp_path / "cb.json")
    cb.lock_file = str(tmp_path / "cb.json.lock")
    
    cb.record_failure("modelC")
    circuit = cb.load()
    assert circuit["modelC"]["failures"] == 1
    
    cb.record_success("modelC")
    circuit = cb.load()
    assert "modelC" not in circuit or circuit["modelC"]["failures"] == 0

import time

def test_circuit_breaker_decay():
    cb = CircuitBreaker("decay_test", max_failures=2, cooldown_seconds=60)
    
    # Clean state
    if os.path.exists(cb.circuit_file):
        os.remove(cb.circuit_file)
        
    import llm_proxy_cli
    
    # Mock time.time to simulate passage of time
    original_time = time.time
    current_mock_time = original_time()
    
    def mock_time():
        return current_mock_time
        
    llm_proxy_cli.time.time = mock_time
    
    try:
        # Failure 1 at T=0
        cb.record_failure("test:model")
        assert cb.check_health("test:model") == True
        
        # Advance time by 400 seconds (past 300s decay threshold)
        current_mock_time += 400
        
        # Failure 2 at T=400
        # This should reset the counter to 0 before adding 1, so it won't trip (max_failures=2)
        cb.record_failure("test:model")
        assert cb.check_health("test:model") == True
        
        # Failure 3 immediately after (T=400)
        # Counter becomes 2 -> trips!
        cb.record_failure("test:model")
        assert cb.check_health("test:model") == False
        
    finally:
        llm_proxy_cli.time.time = original_time
        if os.path.exists(cb.circuit_file):
            os.remove(cb.circuit_file)
