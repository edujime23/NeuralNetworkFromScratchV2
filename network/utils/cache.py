from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Hashable, Self
from inspect import iscoroutinefunction


@dataclass(slots=True)
class DoublyLinkedNode:
    key: Hashable = field(default=None)
    value: Any = field(default=None)
    prev: Self = field(default=None)
    next: Self = field(default=None)


class cache:
    __slots__ = ('max_limit', 'cache', 'head', 'tail', '_hash_args', '_hash_kwargs')

    def __init__(self, max_limit: int = -1):
        self.max_limit = max_limit if max_limit > 0 else float('inf')
        self.cache = {}

        self.head = DoublyLinkedNode()
        self.tail = DoublyLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

        # Pre-compute hash functions to avoid repeated lookups
        self._hash_args = hash
        self._hash_kwargs = lambda d: hash(tuple(sorted(d.items())))

    def _add_to_head(self, node: DoublyLinkedNode) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DoublyLinkedNode) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: DoublyLinkedNode) -> None:
        self._remove_node(node)
        self._add_to_head(node)

    def _pop_tail(self) -> DoublyLinkedNode:
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node

    def _generate_key(self, args: tuple, kwargs: dict) -> int:
        if not kwargs:
            return self._hash_args(args)
        return self._hash_args(args) ^ self._hash_kwargs(kwargs)

    def _handle_cache_hit(self, key: int) -> Any:
        """Common logic for cache hits"""
        node = self.cache[key]
        self._move_to_head(node)
        return node.value

    def _handle_cache_miss(self, key: int, result: Any) -> None:
        """Common logic for cache misses"""
        if len(self.cache) >= self.max_limit:
            tail_node = self._pop_tail()
            del self.cache[tail_node.key]

        new_node = DoublyLinkedNode(key, result)
        self.cache[key] = new_node
        self._add_to_head(new_node)

    def __call__(self, func: Callable) -> Callable:
        is_async = iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                key = self._generate_key(args, kwargs)

                if key in self.cache:
                    return self._handle_cache_hit(key)

                result = await func(*args, **kwargs)
                self._handle_cache_miss(key, result)
                return result

            # Expose API methods for async functions too
            async_wrapper.clear = self.clear
            async_wrapper.get_info = self.get_info
            async_wrapper.get = self.get
            async_wrapper.set = self.set
            async_wrapper.delete = self.delete

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                key = self._generate_key(args, kwargs)

                if key in self.cache:
                    return self._handle_cache_hit(key)

                result = func(*args, **kwargs)
                self._handle_cache_miss(key, result)
                return result

            # Expose API methods
            sync_wrapper.clear = self.clear
            sync_wrapper.get_info = self.get_info
            sync_wrapper.get = self.get
            sync_wrapper.set = self.set
            sync_wrapper.delete = self.delete

            return sync_wrapper

    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get_info(self) -> dict[str, Any]:
        """Get cache statistics"""
        return {
            "current_size": len(self.cache),
            "max_limit": self.max_limit if self.max_limit != float('inf') else -1
        }

    def get(self, key: Hashable) -> Any | None:
        """Get value by key without affecting LRU order"""
        node = self.cache.get(key)
        return node.value if node else None

    def set(self, key: Hashable, value: Any) -> None:
        """Manually set a cache entry"""
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            # Add new
            if len(self.cache) >= self.max_limit:
                tail_node = self._pop_tail()
                del self.cache[tail_node.key]

            new_node = DoublyLinkedNode(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)

    def delete(self, key: Hashable) -> bool:
        """Delete a cache entry, returns True if key existed"""
        if key in self.cache:
            node = self.cache[key]
            self._remove_node(node)
            del self.cache[key]
            return True
        return False