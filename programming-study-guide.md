# Programming Study Guide
**Date:** Friday, October 03, 2025, 11 PM IST

---

## Table of Contents

1. [Sorting Algorithms](#sorting-algorithms)
2. [Queue Data Structure](#queue-data-structure)
3. [Binary Search Tree (BST)](#binary-search-tree-bst)
4. [Heap Sort and Heap Operations](#heap-sort-and-heap-operations)
5. [Binary Tree Traversals](#binary-tree-traversals)
6. [Lowest Common Ancestor (LCA)](#lowest-common-ancestor-lca)
7. [Linked List Operations](#linked-list-operations)
8. [Expression Conversion Algorithms](#expression-conversion-algorithms)
9. [Additional Linked List and Stack Problems](#additional-linked-list-and-stack-problems)
10. [String Algorithms](#string-algorithms)
11. [Divide and Conquer Algorithms](#divide-and-conquer-algorithms)
12. [Sorting Variants](#sorting-variants)
13. [Data Structures Implementations](#data-structures-implementations)
14. [Mathematical Algorithms](#mathematical-algorithms)
15. [Complexity and Optimization Principles](#complexity-and-optimization-principles)

---

## Sorting Algorithms

### Quick Sort

```python
def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]  # Choose rightmost element as pivot
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Usage example
arr = [64, 34, 25, 12, 22, 11, 90]
quicksort(arr, 0, len(arr) - 1)
print("Sorted array:", arr)
```

**Explanation:** Quick Sort uses divide-and-conquer by selecting a pivot element and partitioning the array around it. Elements smaller than the pivot go to the left, larger elements to the right. The algorithm recursively sorts both partitions. The partition function is crucial - it rearranges elements and returns the final position of the pivot.

**Time Complexity:** Best/Average: O(n log n), Worst: O(n²)
**Space Complexity:** O(log n) due to recursion stack
**Use Cases:** General-purpose sorting, when average-case performance is more important than worst-case guarantees.

### Merge Sort

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage example
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = merge_sort(arr)
print("Sorted array:", sorted_arr)
```

**Explanation:** Merge Sort divides the array into halves recursively until single elements remain, then merges them back in sorted order. The merge function combines two sorted arrays into one sorted array by comparing elements from both arrays and selecting the smaller one.

**Time Complexity:** O(n log n) in all cases
**Space Complexity:** O(n) for temporary arrays
**Use Cases:** When stable sorting is required, external sorting for large datasets, linked list sorting.

### Insertion Sort

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key

# Usage example
arr = [64, 34, 25, 12, 22, 11, 90]
insertion_sort(arr)
print("Sorted array:", arr)
```

**Explanation:** Insertion Sort builds the sorted array one element at a time by repeatedly taking elements from the unsorted portion and inserting them into their correct position in the sorted portion. It's similar to how you might sort playing cards in your hand.

**Time Complexity:** Best: O(n), Average/Worst: O(n²)
**Space Complexity:** O(1)
**Use Cases:** Small datasets, nearly sorted data, online algorithms (sorting data as it arrives).

### Selection Sort

```python
def selection_sort(arr):
    n = len(arr)
    
    for i in range(n):
        # Find minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap minimum element with first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

# Usage example
arr = [64, 34, 25, 12, 22, 11, 90]
selection_sort(arr)
print("Sorted array:", arr)
```

**Explanation:** Selection Sort divides the array into sorted and unsorted portions. It repeatedly finds the minimum element from the unsorted portion and places it at the beginning of the unsorted portion. Despite its simplicity, it performs poorly on large datasets.

**Time Complexity:** O(n²) in all cases
**Space Complexity:** O(1)
**Use Cases:** When memory writes are costly, small datasets, when simplicity is preferred over efficiency.

### Bubble Sort

```python
def bubble_sort(arr):
    n = len(arr)
    
    for i in range(n):
        swapped = False
        
        # Last i elements are already sorted
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swapping occurred, array is sorted
        if not swapped:
            break

# Usage example
arr = [64, 34, 25, 12, 22, 11, 90]
bubble_sort(arr)
print("Sorted array:", arr)
```

**Explanation:** Bubble Sort repeatedly steps through the list, compares adjacent elements, and swaps them if they're in the wrong order. The largest elements "bubble" to the end. The optimization with the swapped flag allows early termination if the array becomes sorted before all passes are complete.

**Time Complexity:** Best: O(n), Average/Worst: O(n²)
**Space Complexity:** O(1)
**Use Cases:** Educational purposes, very small datasets, when simplicity is crucial.

### Sorting Algorithms Comparison Table

| Algorithm | Best Time | Average Time | Worst Time | Space Complexity | Stable | In-Place |
|-----------|-----------|--------------|------------|------------------|--------|----------|
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Yes |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | No |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No | Yes |
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Yes |

---

## Queue Data Structure

### Queue Implementation Using Two Stacks

```python
class QueueUsingStacks:
    def __init__(self):
        self.stack1 = []  # For enqueue operations
        self.stack2 = []  # For dequeue operations
    
    def enqueue(self, item):
        """Add item to rear of queue"""
        self.stack1.append(item)
    
    def dequeue(self):
        """Remove and return item from front of queue"""
        if not self.stack2:
            # Transfer all elements from stack1 to stack2
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        if not self.stack2:
            raise IndexError("Queue is empty")
        
        return self.stack2.pop()
    
    def front(self):
        """Return front element without removing"""
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        
        if not self.stack2:
            raise IndexError("Queue is empty")
        
        return self.stack2[-1]
    
    def is_empty(self):
        return len(self.stack1) == 0 and len(self.stack2) == 0
    
    def size(self):
        return len(self.stack1) + len(self.stack2)

# Usage example
queue = QueueUsingStacks()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print("Front:", queue.front())  # Output: 1
print("Dequeue:", queue.dequeue())  # Output: 1
print("Dequeue:", queue.dequeue())  # Output: 2
```

**Explanation:** This implementation uses two stacks to simulate queue behavior. Stack1 handles enqueue operations, while stack2 handles dequeue operations. When dequeuing and stack2 is empty, all elements are transferred from stack1 to stack2, reversing their order to maintain FIFO behavior.

**Time Complexity:** Enqueue: O(1), Dequeue: Amortized O(1), Worst-case O(n)
**Space Complexity:** O(n)
**Use Cases:** Interview questions, understanding stack-queue relationships, systems with limited queue implementations.

---

## Binary Search Tree (BST)

### Node Definition and Basic Operations

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        """Insert a value into the BST"""
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if val < node.val:
            if not node.left:
                node.left = TreeNode(val)
            else:
                self._insert_recursive(node.left, val)
        elif val > node.val:
            if not node.right:
                node.right = TreeNode(val)
            else:
                self._insert_recursive(node.right, val)
        # If val == node.val, we don't insert duplicates
    
    def search(self, val):
        """Search for a value in the BST"""
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_recursive(node.left, val)
        else:
            return self._search_recursive(node.right, val)
    
    def delete(self, val):
        """Delete a value from the BST"""
        self.root = self._delete_recursive(self.root, val)
    
    def _delete_recursive(self, node, val):
        if not node:
            return node
        
        if val < node.val:
            node.left = self._delete_recursive(node.left, val)
        elif val > node.val:
            node.right = self._delete_recursive(node.right, val)
        else:
            # Node to be deleted found
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            
            # Node has two children
            # Find inorder successor (smallest in right subtree)
            min_larger_node = self._find_min(node.right)
            node.val = min_larger_node.val
            node.right = self._delete_recursive(node.right, min_larger_node.val)
        
        return node
    
    def _find_min(self, node):
        """Find minimum value node in a subtree"""
        while node.left:
            node = node.left
        return node
    
    def count_leaf_nodes(self):
        """Count leaf nodes in the BST"""
        return self._count_leaf_recursive(self.root)
    
    def _count_leaf_recursive(self, node):
        if not node:
            return 0
        if not node.left and not node.right:
            return 1
        return (self._count_leaf_recursive(node.left) + 
                self._count_leaf_recursive(node.right))
    
    def count_internal_nodes(self):
        """Count internal nodes in the BST"""
        return self._count_internal_recursive(self.root)
    
    def _count_internal_recursive(self, node):
        if not node or (not node.left and not node.right):
            return 0
        return (1 + self._count_internal_recursive(node.left) + 
                self._count_internal_recursive(node.right))

# Usage example
bst = BST()
values = [50, 30, 70, 20, 40, 60, 80]
for val in values:
    bst.insert(val)

print("Search 40:", bst.search(40))  # Returns node or None
print("Leaf nodes:", bst.count_leaf_nodes())
print("Internal nodes:", bst.count_internal_nodes())
```

**Explanation:** A BST maintains the property that left children are smaller and right children are larger than their parent. Insertion follows this property recursively. Deletion has three cases: no children (simply remove), one child (replace with child), two children (replace with inorder successor). Counting nodes uses recursive traversal.

**Time Complexity:** Search/Insert/Delete: Average O(log n), Worst O(n)
**Space Complexity:** O(h) where h is height, O(log n) average, O(n) worst
**Use Cases:** Efficient searching and sorting, symbol tables, expression parsing.

---

## Heap Sort and Heap Operations

### Heap Implementation and Operations

```python
class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def heapify_up(self, i):
        """Maintain heap property from bottom to top"""
        while i > 0 and self.heap[self.parent(i)] < self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    
    def heapify_down(self, i):
        """Maintain heap property from top to bottom"""
        largest = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right
        
        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self.heapify_down(largest)
    
    def insert(self, val):
        """Insert a new value into the heap"""
        self.heap.append(val)
        self.heapify_up(len(self.heap) - 1)
    
    def extract_max(self):
        """Remove and return the maximum element"""
        if not self.heap:
            raise IndexError("Heap is empty")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()  # Move last element to root
        self.heapify_down(0)
        return max_val
    
    def increase_key(self, i, new_val):
        """Increase value at index i to new_val"""
        if new_val < self.heap[i]:
            raise ValueError("New value is smaller than current value")
        
        self.heap[i] = new_val
        self.heapify_up(i)
    
    def build_heap(self, arr):
        """Build heap from an array"""
        self.heap = arr[:]
        # Start from last non-leaf node and heapify down
        for i in range(len(arr) // 2 - 1, -1, -1):
            self.heapify_down(i)

def heap_sort(arr):
    """Sort array using heap sort algorithm"""
    # Build max heap
    heap = MaxHeap()
    heap.build_heap(arr)
    
    # Extract elements from heap in descending order
    sorted_arr = []
    while heap.heap:
        sorted_arr.append(heap.extract_max())
    
    return sorted_arr[::-1]  # Reverse for ascending order

# Usage example
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = heap_sort(arr)
print("Sorted array:", sorted_arr)

# Heap operations example
heap = MaxHeap()
for val in [10, 20, 15, 30, 40]:
    heap.insert(val)
print("Max element:", heap.extract_max())
heap.increase_key(0, 50)
```

**Explanation:** A max heap is a complete binary tree where each parent is greater than its children. Heapify operations maintain this property. Building a heap from an array takes O(n) time by heapifying from the last non-leaf node upward. Heap sort extracts the maximum element repeatedly, placing it at the end of the array.

**Time Complexity:** Insert/Extract: O(log n), Build Heap: O(n), Heap Sort: O(n log n)
**Space Complexity:** O(1) for heap operations, O(n) for storing heap
**Use Cases:** Priority queues, finding k largest/smallest elements, heap sort algorithm.

---

## Binary Tree Traversals

### Level Order Traversal

```python
from collections import deque

def level_order_traversal(root):
    """Level order traversal using queue"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

**Explanation:** Level order traversal visits nodes level by level from left to right. It uses a queue to process nodes in FIFO order. The outer loop processes one level at a time, while the inner loop processes all nodes in the current level.

### Preorder Traversal

```python
def preorder_recursive(root):
    """Preorder: Root -> Left -> Right (Recursive)"""
    if not root:
        return []
    
    result = [root.val]
    result.extend(preorder_recursive(root.left))
    result.extend(preorder_recursive(root.right))
    return result

def preorder_iterative(root):
    """Preorder: Root -> Left -> Right (Iterative)"""
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first, then left (stack is LIFO)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
```

**Explanation:** Preorder traversal visits root first, then left subtree, then right subtree. The recursive version is straightforward. The iterative version uses a stack and pushes right child before left child to ensure left child is processed first.

### Inorder Traversal

```python
def inorder_recursive(root):
    """Inorder: Left -> Root -> Right (Recursive)"""
    if not root:
        return []
    
    result = []
    result.extend(inorder_recursive(root.left))
    result.append(root.val)
    result.extend(inorder_recursive(root.right))
    return result

def inorder_iterative(root):
    """Inorder: Left -> Root -> Right (Iterative)"""
    result = []
    stack = []
    current = root
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.val)
        
        # Move to right subtree
        current = current.right
    
    return result
```

**Explanation:** Inorder traversal visits left subtree first, then root, then right subtree. For BSTs, this gives sorted order. The iterative version goes to the leftmost node, processes it, then moves to the right subtree.

### Postorder Traversal

```python
def postorder_recursive(root):
    """Postorder: Left -> Right -> Root (Recursive)"""
    if not root:
        return []
    
    result = []
    result.extend(postorder_recursive(root.left))
    result.extend(postorder_recursive(root.right))
    result.append(root.val)
    return result

def postorder_iterative(root):
    """Postorder: Left -> Right -> Root (Iterative)"""
    if not root:
        return []
    
    result = []
    stack = []
    last_visited = None
    current = root
    
    while stack or current:
        if current:
            stack.append(current)
            current = current.left
        else:
            peek_node = stack[-1]
            # If right child exists and hasn't been processed yet
            if peek_node.right and last_visited != peek_node.right:
                current = peek_node.right
            else:
                result.append(peek_node.val)
                last_visited = stack.pop()
    
    return result
```

**Explanation:** Postorder traversal visits left subtree, right subtree, then root. It's useful for deletion operations. The iterative version is more complex, requiring tracking of the last visited node to avoid revisiting the right subtree.

---

## Lowest Common Ancestor (LCA)

### LCA in Generic Binary Tree

```python
def lca_binary_tree(root, p, q):
    """Find LCA in a generic binary tree"""
    if not root or root == p or root == q:
        return root
    
    left = lca_binary_tree(root.left, p, q)
    right = lca_binary_tree(root.right, p, q)
    
    if left and right:
        return root  # Current node is LCA
    
    return left if left else right

# Alternative approach with path storage
def find_path(root, target, path):
    """Find path from root to target node"""
    if not root:
        return False
    
    path.append(root)
    
    if root == target:
        return True
    
    if (find_path(root.left, target, path) or 
        find_path(root.right, target, path)):
        return True
    
    path.pop()  # Backtrack
    return False

def lca_with_paths(root, p, q):
    """Find LCA using path approach"""
    path_p = []
    path_q = []
    
    if not find_path(root, p, path_p) or not find_path(root, q, path_q):
        return None
    
    # Find last common node in paths
    i = 0
    while (i < len(path_p) and i < len(path_q) and 
           path_p[i] == path_q[i]):
        i += 1
    
    return path_p[i - 1] if i > 0 else None
```

### LCA in BST

```python
def lca_bst(root, p, q):
    """Find LCA in Binary Search Tree"""
    if not root:
        return None
    
    # If both nodes are smaller, LCA is in left subtree
    if p.val < root.val and q.val < root.val:
        return lca_bst(root.left, p, q)
    
    # If both nodes are larger, LCA is in right subtree
    if p.val > root.val and q.val > root.val:
        return lca_bst(root.right, p, q)
    
    # If one is smaller and one is larger, current node is LCA
    return root

def lca_bst_iterative(root, p, q):
    """Iterative LCA in BST"""
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
    return None
```

### LCA with Parent Pointers

```python
class TreeNodeWithParent:
    def __init__(self, val=0, left=None, right=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent

def lca_with_parent_pointers(p, q):
    """Find LCA when nodes have parent pointers"""
    # Get depths of both nodes
    def get_depth(node):
        depth = 0
        while node.parent:
            depth += 1
            node = node.parent
        return depth
    
    depth_p = get_depth(p)
    depth_q = get_depth(q)
    
    # Bring both nodes to same level
    while depth_p > depth_q:
        p = p.parent
        depth_p -= 1
    
    while depth_q > depth_p:
        q = q.parent
        depth_q -= 1
    
    # Move both nodes up until they meet
    while p != q:
        p = p.parent
        q = q.parent
    
    return p
```

**Explanation:** LCA algorithms find the deepest node that is an ancestor of both given nodes. In generic binary trees, we use recursion to search both subtrees. In BSTs, we can use the ordering property to decide which direction to search. With parent pointers, we can move up from both nodes until they meet.

**Time Complexity:** Generic: O(n), BST: O(h), Parent pointers: O(h)
**Space Complexity:** Generic: O(h) recursion, BST: O(1) iterative, Parent: O(1)
**Use Cases:** Tree queries, finding relationships between nodes, phylogenetic analysis.

---

## Linked List Operations

### Basic Node Class and Operations

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.val)
            current = current.next
        return elements
```

### Cycle Detection (Floyd's Algorithm)

```python
def has_cycle(head):
    """Detect cycle using Floyd's cycle detection algorithm"""
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    # Move slow one step, fast two steps
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

def find_cycle_start(head):
    """Find the starting node of the cycle"""
    if not head or not head.next:
        return None
    
    # First, detect if cycle exists
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle found
    
    # Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

**Explanation:** Floyd's algorithm uses two pointers moving at different speeds. If there's a cycle, the fast pointer will eventually meet the slow pointer. To find the cycle start, reset one pointer to head and move both at the same speed until they meet.

### Finding Middle Node

```python
def find_middle(head):
    """Find middle node using two-pointer technique"""
    if not head:
        return None
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

def find_middle_with_length(head):
    """Alternative: find middle by calculating length"""
    if not head:
        return None
    
    # Calculate length
    length = 0
    current = head
    while current:
        length += 1
        current = current.next
    
    # Find middle
    middle_index = length // 2
    current = head
    for _ in range(middle_index):
        current = current.next
    
    return current
```

### Reversing Linked List

```python
def reverse_iterative(head):
    """Reverse linked list iteratively"""
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev

def reverse_recursive(head):
    """Reverse linked list recursively"""
    if not head or not head.next:
        return head
    
    reversed_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    
    return reversed_head
```

### Reversing Between Two Positions

```python
def reverse_between(head, left, right):
    """Reverse linked list between positions left and right"""
    if not head or left == right:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    # Move to position before left
    for _ in range(left - 1):
        prev = prev.next
    
    # Start reversing from left to right
    current = prev.next
    
    for _ in range(right - left):
        next_temp = current.next
        current.next = next_temp.next
        next_temp.next = prev.next
        prev.next = next_temp
    
    return dummy.next
```

### Reversing in Groups of K

```python
def reverse_k_group(head, k):
    """Reverse linked list in groups of k"""
    def get_length(node):
        length = 0
        while node:
            length += 1
            node = node.next
        return length
    
    def reverse_group(start, k):
        prev = None
        current = start
        
        for _ in range(k):
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev, current
    
    length = get_length(head)
    if length < k:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    prev_group_end = dummy
    
    while length >= k:
        group_start = prev_group_end.next
        new_group_start, next_group_start = reverse_group(group_start, k)
        
        prev_group_end.next = new_group_start
        group_start.next = next_group_start
        prev_group_end = group_start
        
        length -= k
    
    return dummy.next
```

**Explanation:** Linked list operations often use the two-pointer technique. Cycle detection uses different speeds to catch cycles. Reversing manipulates next pointers systematically. Group reversal combines individual reversal with careful pointer management to maintain list integrity.

**Time Complexity:** Most operations: O(n), Space: O(1) iterative, O(n) recursive
**Use Cases:** Memory-efficient data storage, undo operations, music playlists, browser history.

---

## Expression Conversion Algorithms

### Operator Precedence and Associativity

```python
def get_precedence(op):
    """Return precedence of operator"""
    precedences = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    return precedences.get(op, 0)

def is_right_associative(op):
    """Check if operator is right associative"""
    return op == '^'

def is_operator(char):
    """Check if character is an operator"""
    return char in ['+', '-', '*', '/', '^']

def is_operand(char):
    """Check if character is an operand"""
    return char.isalnum()
```

### Infix to Postfix Conversion

```python
def infix_to_postfix(infix):
    """Convert infix expression to postfix"""
    result = []
    stack = []
    
    for char in infix:
        if is_operand(char):
            result.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            # Pop until opening parenthesis
            while stack and stack[-1] != '(':
                result.append(stack.pop())
            stack.pop()  # Remove '('
        elif is_operator(char):
            # Pop operators with higher or equal precedence
            while (stack and stack[-1] != '(' and
                   (get_precedence(stack[-1]) > get_precedence(char) or
                    (get_precedence(stack[-1]) == get_precedence(char) and
                     not is_right_associative(char)))):
                result.append(stack.pop())
            stack.append(char)
    
    # Pop remaining operators
    while stack:
        result.append(stack.pop())
    
    return ''.join(result)

# Test example
infix_expr = "a+b*c-d"
postfix_expr = infix_to_postfix(infix_expr)
print(f"Infix: {infix_expr}")
print(f"Postfix: {postfix_expr}")  # Output: abc*+d-
```

### Infix to Prefix Conversion

```python
def infix_to_prefix(infix):
    """Convert infix expression to prefix"""
    # Reverse the infix expression
    reversed_infix = infix[::-1]
    
    # Replace ( with ) and vice versa
    temp = []
    for char in reversed_infix:
        if char == '(':
            temp.append(')')
        elif char == ')':
            temp.append('(')
        else:
            temp.append(char)
    
    reversed_infix = ''.join(temp)
    
    # Get postfix of modified expression
    result = []
    stack = []
    
    for char in reversed_infix:
        if is_operand(char):
            result.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                result.append(stack.pop())
            stack.pop()
        elif is_operator(char):
            # For prefix, we need to consider right associativity differently
            while (stack and stack[-1] != '(' and
                   (get_precedence(stack[-1]) > get_precedence(char) or
                    (get_precedence(stack[-1]) == get_precedence(char) and
                     is_right_associative(char)))):
                result.append(stack.pop())
            stack.append(char)
    
    while stack:
        result.append(stack.pop())
    
    # Reverse the result to get prefix
    return ''.join(result[::-1])
```

### Postfix to Infix Conversion

```python
def postfix_to_infix(postfix):
    """Convert postfix expression to infix"""
    stack = []
    
    for char in postfix:
        if is_operand(char):
            stack.append(char)
        elif is_operator(char):
            # Pop two operands
            operand2 = stack.pop()
            operand1 = stack.pop()
            
            # Create infix expression
            expression = f"({operand1}{char}{operand2})"
            stack.append(expression)
    
    return stack[0]

# Test example
postfix_expr = "abc*+d-"
infix_expr = postfix_to_infix(postfix_expr)
print(f"Postfix: {postfix_expr}")
print(f"Infix: {infix_expr}")  # Output: ((a+(b*c))-d)
```

**Explanation:** Expression conversion algorithms use stacks to manage operator precedence and associativity. The Shunting Yard algorithm (infix to postfix) processes operators based on their precedence. Prefix conversion reverses the input and modifies the algorithm. These conversions are fundamental in compiler design and expression evaluation.

**Time Complexity:** O(n) for all conversions
**Space Complexity:** O(n) for stack storage
**Use Cases:** Compiler design, calculator applications, expression parsing, syntax analysis.

---

## Additional Linked List and Stack Problems

### Merging Two Sorted Linked Lists

```python
def merge_two_sorted_iterative(l1, l2):
    """Merge two sorted linked lists iteratively"""
    dummy = ListNode(0)
    current = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    
    # Append remaining nodes
    current.next = l1 if l1 else l2
    
    return dummy.next

def merge_two_sorted_recursive(l1, l2):
    """Merge two sorted linked lists recursively"""
    if not l1:
        return l2
    if not l2:
        return l1
    
    if l1.val <= l2.val:
        l1.next = merge_two_sorted_recursive(l1.next, l2)
        return l1
    else:
        l2.next = merge_two_sorted_recursive(l1, l2.next)
        return l2
```

### Removing Nth Node from End

```python
def remove_nth_from_end_two_pass(head, n):
    """Remove nth node from end using two passes"""
    # First pass: calculate length
    length = 0
    current = head
    while current:
        length += 1
        current = current.next
    
    # Handle edge case: remove first node
    if length == n:
        return head.next
    
    # Second pass: find node to remove
    current = head
    for _ in range(length - n - 1):
        current = current.next
    
    current.next = current.next.next
    return head

def remove_nth_from_end_one_pass(head, n):
    """Remove nth node from end using one pass (two pointers)"""
    dummy = ListNode(0)
    dummy.next = head
    first = dummy
    second = dummy
    
    # Move first pointer n+1 steps ahead
    for _ in range(n + 1):
        first = first.next
    
    # Move both pointers until first reaches end
    while first:
        first = first.next
        second = second.next
    
    # Remove the nth node
    second.next = second.next.next
    
    return dummy.next
```

### Intersection of Two Linked Lists

```python
def get_intersection_hash_set(headA, headB):
    """Find intersection using hash set"""
    visited = set()
    
    # Add all nodes from first list to set
    current = headA
    while current:
        visited.add(current)
        current = current.next
    
    # Check if any node from second list is in set
    current = headB
    while current:
        if current in visited:
            return current
        current = current.next
    
    return None

def get_intersection_two_pointers(headA, headB):
    """Find intersection using two pointers"""
    if not headA or not headB:
        return None
    
    pointerA = headA
    pointerB = headB
    
    # When one pointer reaches end, redirect to other list's head
    while pointerA != pointerB:
        pointerA = pointerA.next if pointerA else headB
        pointerB = pointerB.next if pointerB else headA
    
    return pointerA  # Either intersection node or None
```

### Valid Parentheses Problems

```python
def is_valid_parentheses(s):
    """Check if parentheses are valid"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

def longest_valid_parentheses(s):
    """Find length of longest valid parentheses substring"""
    stack = [-1]  # Initialize with -1 for base case
    max_length = 0
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:  # char == ')'
            stack.pop()
            if not stack:
                stack.append(i)  # No matching '(' found
            else:
                max_length = max(max_length, i - stack[-1])
    
    return max_length

def min_remove_to_make_valid(s):
    """Minimum removals to make parentheses valid"""
    stack = []
    to_remove = set()
    
    # Find unmatched parentheses
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                to_remove.add(i)
    
    # Add unmatched opening parentheses
    to_remove.update(stack)
    
    # Build result string
    result = []
    for i, char in enumerate(s):
        if i not in to_remove:
            result.append(char)
    
    return ''.join(result)
```

### Randomized QuickSort Variants

```python
import random

def randomized_quicksort_random_pivot(arr, low, high):
    """QuickSort with random pivot selection"""
    if low < high:
        # Choose random pivot and swap with last element
        random_index = random.randint(low, high)
        arr[random_index], arr[high] = arr[high], arr[random_index]
        
        pi = partition(arr, low, high)
        randomized_quicksort_random_pivot(arr, low, pi - 1)
        randomized_quicksort_random_pivot(arr, pi + 1, high)

def randomized_quicksort_shuffle(arr):
    """QuickSort with initial array shuffling"""
    # Shuffle array first
    for i in range(len(arr)):
        j = random.randint(i, len(arr) - 1)
        arr[i], arr[j] = arr[j], arr[i]
    
    quicksort(arr, 0, len(arr) - 1)

def three_way_partition_quicksort(arr, low, high):
    """QuickSort with 3-way partitioning for duplicate elements"""
    if low < high:
        lt, gt = three_way_partition(arr, low, high)
        three_way_partition_quicksort(arr, low, lt - 1)
        three_way_partition_quicksort(arr, gt + 1, high)

def three_way_partition(arr, low, high):
    """Partition into <pivot, =pivot, >pivot regions"""
    pivot = arr[low]
    i = low
    lt = low  # arr[low..lt-1] < pivot
    gt = high + 1  # arr[gt..high] > pivot
    
    while i < gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            gt -= 1
            arr[i], arr[gt] = arr[gt], arr[i]
        else:
            i += 1
    
    return lt, gt
```

**Explanation:** These advanced problems combine multiple concepts. Merging sorted lists maintains order while combining. Removing nth from end uses the two-pointer technique with careful distance management. Intersection problems leverage the constraint that nodes after intersection are shared. Parentheses problems use stacks to match opening and closing brackets. Randomized quicksort variants improve average-case performance by avoiding worst-case pivot selections.

---

## String Algorithms

### Palindrome Checking

```python
def is_palindrome_simple(s):
    """Simple palindrome check"""
    return s == s[::-1]

def is_palindrome_two_pointers(s):
    """Palindrome check using two pointers"""
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    
    return True

def is_palindrome_alphanumeric(s):
    """Check palindrome considering only alphanumeric characters"""
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]

def longest_palindromic_substring(s):
    """Find longest palindromic substring"""
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    start = 0
    max_len = 0
    
    for i in range(len(s)):
        # Check for odd-length palindromes
        len1 = expand_around_center(i, i)
        # Check for even-length palindromes
        len2 = expand_around_center(i, i + 1)
        
        current_max = max(len1, len2)
        if current_max > max_len:
            max_len = current_max
            start = i - (current_max - 1) // 2
    
    return s[start:start + max_len]
```

### String Reversal

```python
def reverse_string_builtin(s):
    """Reverse string using built-in method"""
    return s[::-1]

def reverse_string_iterative(s):
    """Reverse string iteratively"""
    chars = list(s)
    left, right = 0, len(chars) - 1
    
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    
    return ''.join(chars)

def reverse_string_recursive(s):
    """Reverse string recursively"""
    if len(s) <= 1:
        return s
    return reverse_string_recursive(s[1:]) + s[0]

def reverse_words_in_string(s):
    """Reverse words in a string"""
    words = s.split()
    return ' '.join(reversed(words))
```

### Pattern Matching (KMP Algorithm)

```python
def kmp_search(text, pattern):
    """Knuth-Morris-Pratt pattern searching algorithm"""
    def compute_lps(pattern):
        """Compute Longest Proper Prefix which is also Suffix array"""
        m = len(pattern)
        lps = [0] * m
        length = 0  # Length of previous longest prefix suffix
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    n = len(text)
    m = len(pattern)
    
    if m == 0:
        return []
    
    lps = compute_lps(pattern)
    matches = []
    
    i = 0  # Index for text
    j = 0  # Index for pattern
    
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

# Usage example
text = "ABABDABACDABABCABCABCABCABC"
pattern = "ABABCABCABCABC"
matches = kmp_search(text, pattern)
print(f"Pattern found at indices: {matches}")
```

**Explanation:** String algorithms handle text processing efficiently. Palindrome checking can be optimized using two pointers to avoid creating reversed strings. The KMP algorithm uses preprocessing to create a failure function that helps skip characters during pattern matching, achieving O(n + m) time complexity instead of the naive O(nm) approach.

**Time Complexity:** Palindrome: O(n), KMP: O(n + m)
**Space Complexity:** Palindrome: O(1), KMP: O(m) for LPS array
**Use Cases:** Text editors, search engines, DNA sequence analysis, data validation.

---

## Divide and Conquer Algorithms

### Binary Search

```python
def binary_search_iterative(arr, target):
    """Binary search iterative implementation"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_recursive(arr, target, left, right):
    """Binary search recursive implementation"""
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def binary_search_leftmost(arr, target):
    """Find leftmost occurrence of target"""
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def binary_search_rightmost(arr, target):
    """Find rightmost occurrence of target"""
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### QuickSelect Algorithm

```python
def quickselect(arr, k):
    """Find k-th smallest element using QuickSelect"""
    if not arr:
        return None
    
    def quickselect_helper(arr, left, right, k):
        if left == right:
            return arr[left]
        
        # Partition around random pivot
        pivot_index = random.randint(left, right)
        arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
        
        # Partition
        pivot = partition_quickselect(arr, left, right)
        
        if k == pivot:
            return arr[k]
        elif k < pivot:
            return quickselect_helper(arr, left, pivot - 1, k)
        else:
            return quickselect_helper(arr, pivot + 1, right, k)
    
    return quickselect_helper(arr[:], 0, len(arr) - 1, k - 1)  # k-1 for 0-based indexing

def partition_quickselect(arr, left, right):
    """Partition function for QuickSelect"""
    pivot = arr[right]
    i = left
    
    for j in range(left, right):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    
    arr[i], arr[right] = arr[right], arr[i]
    return i

def find_kth_largest(arr, k):
    """Find k-th largest element"""
    return quickselect(arr, len(arr) - k + 1)

def find_median(arr):
    """Find median using QuickSelect"""
    n = len(arr)
    if n % 2 == 1:
        return quickselect(arr, (n + 1) // 2)
    else:
        left_median = quickselect(arr, n // 2)
        right_median = quickselect(arr, n // 2 + 1)
        return (left_median + right_median) / 2.0

# Usage example
arr = [3, 2, 1, 5, 6, 4]
print(f"2nd smallest: {quickselect(arr, 2)}")  # Output: 2
print(f"2nd largest: {find_kth_largest(arr, 2)}")  # Output: 5
```

**Explanation:** Divide and conquer algorithms break problems into smaller subproblems. Binary search repeatedly divides the search space in half, requiring sorted input. QuickSelect uses partitioning like QuickSort but only recurses on one side, making it efficient for finding order statistics.

**Time Complexity:** Binary Search: O(log n), QuickSelect: Average O(n), Worst O(n²)
**Space Complexity:** Binary Search: O(1) iterative/O(log n) recursive, QuickSelect: O(log n)
**Use Cases:** Searching sorted data, finding medians, top-k problems, database indexing.

---

## Sorting Variants

### Counting Sort

```python
def counting_sort(arr, max_val=None):
    """Counting sort for non-negative integers"""
    if not arr:
        return arr
    
    # Find maximum value if not provided
    if max_val is None:
        max_val = max(arr)
    
    # Initialize count array
    count = [0] * (max_val + 1)
    
    # Count occurrences
    for num in arr:
        count[num] += 1
    
    # Reconstruct sorted array
    result = []
    for i in range(len(count)):
        result.extend([i] * count[i])
    
    return result

def counting_sort_stable(arr, max_val=None):
    """Stable version of counting sort"""
    if not arr:
        return arr
    
    if max_val is None:
        max_val = max(arr)
    
    count = [0] * (max_val + 1)
    output = [0] * len(arr)
    
    # Count occurrences
    for num in arr:
        count[num] += 1
    
    # Calculate cumulative count
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    # Build output array (traverse from right to maintain stability)
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1
    
    return output

# Usage example
arr = [4, 2, 2, 8, 3, 3, 1]
sorted_arr = counting_sort(arr)
print("Counting sort result:", sorted_arr)
```

### Radix Sort

```python
def radix_sort(arr):
    """Radix sort for non-negative integers"""
    if not arr:
        return arr
    
    def counting_sort_by_digit(arr, exp):
        """Counting sort based on digit represented by exp"""
        n = len(arr)
        output = [0] * n
        count = [0] * 10  # 10 possible digits (0-9)
        
        # Count occurrences of each digit
        for num in arr:
            index = (num // exp) % 10
            count[index] += 1
        
        # Calculate cumulative count
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        # Build output array
        for i in range(n - 1, -1, -1):
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
        
        return output
    
    # Find maximum number to know number of digits
    max_num = max(arr)
    
    # Do counting sort for every digit
    exp = 1  # Start with least significant digit
    while max_num // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr

def radix_sort_negative(arr):
    """Radix sort that handles negative numbers"""
    if not arr:
        return arr
    
    # Separate positive and negative numbers
    positive = [x for x in arr if x >= 0]
    negative = [-x for x in arr if x < 0]
    
    # Sort positive numbers normally
    sorted_positive = radix_sort(positive) if positive else []
    
    # Sort negative numbers and reverse order
    sorted_negative = radix_sort(negative) if negative else []
    sorted_negative = [-x for x in reversed(sorted_negative)]
    
    return sorted_negative + sorted_positive

# Usage example
arr = [170, 45, 75, 90, 2, 802, 24, 66]
sorted_arr = radix_sort(arr)
print("Radix sort result:", sorted_arr)
```

**Explanation:** Counting sort works by counting occurrences of each element and reconstructing the sorted array. It's efficient for integers with a small range. Radix sort applies counting sort to each digit position, starting from the least significant digit. Both are non-comparison based sorts with linear time complexity under certain conditions.

**Time Complexity:** Counting Sort: O(n + k), Radix Sort: O(d × (n + k))
**Space Complexity:** Counting Sort: O(k), Radix Sort: O(n + k)
**Use Cases:** Sorting integers with limited range, external sorting, parallel processing.

---

## Data Structures Implementations

### Stack Implementation

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        """Return top item without removing"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
    
    def __str__(self):
        return str(self.items)

# Usage example
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print("Stack:", stack)
print("Pop:", stack.pop())
print("Peek:", stack.peek())
```

### Queue Implementation

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        """Add item to rear of queue"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()
    
    def front(self):
        """Return front item without removing"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

class CircularQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = [None] * max_size
        self.front = 0
        self.rear = 0
        self.count = 0
    
    def enqueue(self, item):
        if self.is_full():
            raise OverflowError("Queue is full")
        
        self.queue[self.rear] = item
        self.rear = (self.rear + 1) % self.max_size
        self.count += 1
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.max_size
        self.count -= 1
        return item
    
    def is_empty(self):
        return self.count == 0
    
    def is_full(self):
        return self.count == self.max_size
    
    def size(self):
        return self.count
```

### Deque (Double-ended Queue)

```python
class Deque:
    def __init__(self):
        self.items = []
    
    def add_front(self, item):
        """Add item to front"""
        self.items.insert(0, item)
    
    def add_rear(self, item):
        """Add item to rear"""
        self.items.append(item)
    
    def remove_front(self):
        """Remove item from front"""
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items.pop(0)
    
    def remove_rear(self):
        """Remove item from rear"""
        if self.is_empty():
            raise IndexError("Deque is empty")
        return self.items.pop()
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

# More efficient deque using doubly linked list
class Node:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class EfficientDeque:
    def __init__(self):
        # Create dummy head and tail nodes
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.count = 0
    
    def add_front(self, val):
        node = Node(val, self.head, self.head.next)
        self.head.next.prev = node
        self.head.next = node
        self.count += 1
    
    def add_rear(self, val):
        node = Node(val, self.tail.prev, self.tail)
        self.tail.prev.next = node
        self.tail.prev = node
        self.count += 1
    
    def remove_front(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        
        node = self.head.next
        node.prev.next = node.next
        node.next.prev = node.prev
        self.count -= 1
        return node.val
    
    def remove_rear(self):
        if self.is_empty():
            raise IndexError("Deque is empty")
        
        node = self.tail.prev
        node.prev.next = node.next
        node.next.prev = node.prev
        self.count -= 1
        return node.val
    
    def is_empty(self):
        return self.count == 0
    
    def size(self):
        return self.count
```

### Linked List Variants

```python
class DoublyLinkedListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, val):
        new_node = DoublyLinkedListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def prepend(self, val):
        new_node = DoublyLinkedListNode(val)
        
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def delete(self, val):
        current = self.head
        
        while current:
            if current.val == val:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                
                self.size -= 1
                return True
            
            current = current.next
        
        return False

class CircularLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, val):
        new_node = ListNode(val)
        
        if not self.head:
            self.head = new_node
            new_node.next = self.head  # Point to itself
        else:
            # Find last node
            current = self.head
            while current.next != self.head:
                current = current.next
            
            current.next = new_node
            new_node.next = self.head
        
        self.size += 1
    
    def display(self):
        if not self.head:
            return []
        
        elements = []
        current = self.head
        
        while True:
            elements.append(current.val)
            current = current.next
            if current == self.head:
                break
        
        return elements
```

**Explanation:** These implementations provide the fundamental building blocks for more complex data structures. Stacks follow LIFO principle, queues follow FIFO. Circular queues efficiently use fixed-size arrays. Deques allow insertion/deletion from both ends. Doubly linked lists enable bidirectional traversal, while circular lists have no null terminations.

**Time Complexity:** Most operations: O(1), Search/Delete by value: O(n)
**Space Complexity:** O(n) for storing elements
**Use Cases:** Expression evaluation, BFS/DFS traversal, undo operations, LRU caches.

---

## Mathematical Algorithms

### Euclidean Algorithm for GCD

```python
def gcd_recursive(a, b):
    """Greatest Common Divisor using recursion"""
    if b == 0:
        return a
    return gcd_recursive(b, a % b)

def gcd_iterative(a, b):
    """Greatest Common Divisor using iteration"""
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    """Extended Euclidean Algorithm"""
    if a == 0:
        return b, 0, 1
    
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return gcd, x, y

def lcm(a, b):
    """Least Common Multiple"""
    return abs(a * b) // gcd_iterative(a, b)

def gcd_multiple(numbers):
    """GCD of multiple numbers"""
    result = numbers[0]
    for i in range(1, len(numbers)):
        result = gcd_iterative(result, numbers[i])
        if result == 1:
            break  # Early termination
    return result

# Usage example
print("GCD(48, 18):", gcd_iterative(48, 18))  # Output: 6
print("LCM(48, 18):", lcm(48, 18))  # Output: 144
gcd_val, x, y = extended_gcd(30, 18)
print(f"Extended GCD: {gcd_val} = 30*{x} + 18*{y}")
```

### Sieve of Eratosthenes

```python
def sieve_of_eratosthenes(n):
    """Find all prime numbers up to n"""
    if n < 2:
        return []
    
    # Initialize sieve array
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    # Sieve process
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark multiples of i as composite
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    # Collect prime numbers
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    return primes

def segmented_sieve(n):
    """Segmented sieve for large numbers"""
    import math
    
    limit = int(math.sqrt(n)) + 1
    primes = sieve_of_eratosthenes(limit)
    
    # Process segments
    segment_size = limit
    all_primes = primes[:]
    
    for low in range(limit, n + 1, segment_size):
        high = min(low + segment_size - 1, n)
        
        # Create segment array
        segment = [True] * (high - low + 1)
        
        # Mark composites in current segment
        for prime in primes:
            # Find first multiple of prime in segment
            start = max(prime * prime, (low + prime - 1) // prime * prime)
            
            for j in range(start, high + 1, prime):
                segment[j - low] = False
        
        # Add primes from current segment
        for i in range(len(segment)):
            if segment[i]:
                all_primes.append(low + i)
    
    return all_primes

def prime_factorization(n):
    """Find prime factorization of n"""
    factors = []
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    
    if n > 1:
        factors.append(n)
    
    return factors

# Usage example
primes = sieve_of_eratosthenes(30)
print("Primes up to 30:", primes)
print("Prime factorization of 60:", prime_factorization(60))
```

### Modular Exponentiation

```python
def power_iterative(base, exp, mod=None):
    """Fast exponentiation using iteration"""
    result = 1
    base = base % mod if mod else base
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod if mod else result * base
        
        exp = exp >> 1  # Divide by 2
        base = (base * base) % mod if mod else base * base
    
    return result

def power_recursive(base, exp, mod=None):
    """Fast exponentiation using recursion"""
    if exp == 0:
        return 1
    
    if exp % 2 == 0:
        half_power = power_recursive(base, exp // 2, mod)
        result = half_power * half_power
    else:
        result = base * power_recursive(base, exp - 1, mod)
    
    return result % mod if mod else result

def modular_inverse(a, m):
    """Find modular inverse using extended Euclidean algorithm"""
    def extended_gcd_for_inverse(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd_for_inverse(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, y = extended_gcd_for_inverse(a % m, m)
    
    if gcd != 1:
        raise ValueError("Modular inverse does not exist")
    
    return (x % m + m) % m

def chinese_remainder_theorem(remainders, moduli):
    """Solve system of congruences using CRT"""
    total = 0
    prod = 1
    
    for m in moduli:
        prod *= m
    
    for r, m in zip(remainders, moduli):
        p = prod // m
        total += r * modular_inverse(p, m) * p
    
    return total % prod

# Usage example
print("2^10 mod 1000:", power_iterative(2, 10, 1000))
print("Modular inverse of 3 mod 11:", modular_inverse(3, 11))

# CRT example: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
remainders = [2, 3, 2]
moduli = [3, 5, 7]
solution = chinese_remainder_theorem(remainders, moduli)
print("CRT solution:", solution)
```

**Explanation:** Mathematical algorithms form the foundation of many computational problems. The Euclidean algorithm efficiently finds GCD using the principle that gcd(a,b) = gcd(b, a mod b). The Sieve of Eratosthenes systematically eliminates composites to find primes. Modular exponentiation uses repeated squaring to compute large powers efficiently, crucial in cryptography.

**Time Complexity:** GCD: O(log min(a,b)), Sieve: O(n log log n), Modular Exp: O(log exp)
**Space Complexity:** GCD: O(1), Sieve: O(n), Modular Exp: O(1)
**Use Cases:** Cryptography, number theory, competitive programming, computer algebra.

---

## Complexity and Optimization Principles

### Time and Space Complexity Basics

**Time Complexity Analysis:**

1. **Constant Time - O(1)**: Operations that take the same time regardless of input size
   ```python
   def get_first_element(arr):
       return arr[0] if arr else None  # O(1)
   ```

2. **Logarithmic Time - O(log n)**: Divides problem size in half each step
   ```python
   def binary_search_example(arr, target):
       # Each comparison eliminates half the search space
       # Height of binary tree: log₂(n)
       pass
   ```

3. **Linear Time - O(n)**: Processes each element once
   ```python
   def find_maximum(arr):
       max_val = arr[0]
       for val in arr:  # O(n)
           if val > max_val:
               max_val = val
       return max_val
   ```

4. **Linearithmic Time - O(n log n)**: Combines linear and logarithmic
   ```python
   def merge_sort_analysis(arr):
       # log n levels, each level processes n elements
       # Common in efficient sorting algorithms
       pass
   ```

5. **Quadratic Time - O(n²)**: Nested loops over input
   ```python
   def bubble_sort_analysis(arr):
       for i in range(len(arr)):      # O(n)
           for j in range(len(arr)):  # O(n)
               # O(n²) total
               pass
   ```

**Space Complexity Analysis:**

1. **Constant Space - O(1)**: Fixed memory usage
   ```python
   def swap_variables(a, b):
       # Only uses a fixed amount of extra memory
       temp = a
       a = b
       b = temp
   ```

2. **Linear Space - O(n)**: Memory grows with input
   ```python
   def create_copy(arr):
       return arr[:]  # Creates new array of size n
   ```

3. **Logarithmic Space - O(log n)**: Recursive call stack
   ```python
   def binary_search_recursive(arr, target, left, right):
       # Maximum recursion depth: log n
       if left > right:
           return -1
       mid = (left + right) // 2
       if arr[mid] == target:
           return mid
       elif arr[mid] < target:
           return binary_search_recursive(arr, target, mid + 1, right)
       else:
           return binary_search_recursive(arr, target, left, mid - 1)
   ```

### Optimization Techniques

#### 1. Two-Pointer Technique
```python
def optimized_two_sum(arr, target):
    """Find two numbers that sum to target in sorted array"""
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

# Naive O(n²) vs Optimized O(n)
def naive_two_sum(arr, target):
    """Naive approach - O(n²)"""
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == target:
                return [i, j]
    return []
```

#### 2. Sliding Window Technique
```python
def max_sum_subarray_fixed_size(arr, k):
    """Find maximum sum of subarray of size k"""
    if len(arr) < k:
        return 0
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide window: remove first element, add next element
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

def longest_substring_without_repeating(s):
    """Find longest substring without repeating characters"""
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Shrink window until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

#### 3. Memoization and Dynamic Programming
```python
def fibonacci_optimized(n, memo={}):
    """Fibonacci with memoization"""
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_optimized(n - 1, memo) + fibonacci_optimized(n - 2, memo)
    return memo[n]

def fibonacci_bottom_up(n):
    """Bottom-up DP approach"""
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

# Space optimization: O(n) -> O(1)
def fibonacci_space_optimized(n):
    """Space-optimized version"""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b
```

#### 4. Early Termination and Pruning
```python
def optimized_search(arr, target):
    """Search with early termination"""
    for i, val in enumerate(arr):
        if val == target:
            return i
        if val > target and arr == sorted(arr):
            break  # Early termination for sorted array
    return -1

def prime_check_optimized(n):
    """Optimized primality test"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Only check divisors of form 6k ± 1
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True
```

### Writing Efficient Code

#### Iterative vs Recursive Guidelines

**Use Iterative When:**
- Memory is a constraint (avoid stack overflow)
- Performance is critical
- Problem has simple iterative solution

**Use Recursive When:**
- Problem has natural recursive structure (trees, divide-and-conquer)
- Code readability is important
- Tail recursion can be optimized

```python
# Example: Tree traversal comparison
def tree_height_recursive(root):
    """Recursive approach - cleaner but uses O(h) space"""
    if not root:
        return 0
    return 1 + max(tree_height_recursive(root.left), 
                   tree_height_recursive(root.right))

def tree_height_iterative(root):
    """Iterative approach - O(1) extra space worst case"""
    if not root:
        return 0
    
    height = 0
    queue = [(root, 1)]
    
    while queue:
        node, level = queue.pop(0)
        height = max(height, level)
        
        if node.left:
            queue.append((node.left, level + 1))
        if node.right:
            queue.append((node.right, level + 1))
    
    return height
```

#### Common Optimization Patterns

1. **Cache Frequently Computed Values**
2. **Use Appropriate Data Structures** (hash tables for O(1) lookup)
3. **Avoid Unnecessary Computations** (break early, skip redundant work)
4. **Space-Time Tradeoffs** (use extra memory to save time when beneficial)
5. **Batch Operations** (process multiple items together)
6. **Use Built-in Functions** (often optimized in implementation)

**Complexity Summary Table:**

| Operation | Best Case | Average Case | Worst Case | Space |
|-----------|-----------|--------------|------------|-------|
| Array Access | O(1) | O(1) | O(1) | O(1) |
| Array Search | O(1) | O(n) | O(n) | O(1) |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |
| Hash Table Lookup | O(1) | O(1) | O(n) | O(n) |
| BST Operations | O(log n) | O(log n) | O(n) | O(n) |
| Heap Operations | O(1) | O(log n) | O(log n) | O(n) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |

---

## Conclusion

This comprehensive study guide covers essential data structures and algorithms with both theoretical understanding and practical implementation. Each algorithm includes complexity analysis, use cases, and implementation nuances critical for technical interviews and competitive programming.

**Key Takeaways:**
1. **Understand the fundamentals** - Master basic operations before moving to advanced topics
2. **Practice implementation** - Write code from scratch to understand nuances
3. **Analyze complexity** - Always consider time and space tradeoffs
4. **Choose appropriate tools** - Different problems require different approaches
5. **Optimize incrementally** - Start with working solution, then optimize

**Study Strategy:**
- Implement each algorithm multiple times
- Trace through examples step by step
- Compare different approaches for the same problem
- Practice on coding platforms with various test cases
- Review complexity analysis for each implementation

Remember: The goal isn't just to memorize implementations, but to understand when and why to use each algorithm. This foundation will help you tackle new problems and optimize solutions effectively.

---

**Document prepared on:** Friday, October 03, 2025, 11 PM IST  
**Total algorithms covered:** 50+ implementations  
**Sections:** 15 major topics with comprehensive examples and explanations