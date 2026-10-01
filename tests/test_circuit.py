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
