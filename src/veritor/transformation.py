"""
StableHLO transformation utilities for Veritor.

This module provides graph transformations for StableHLO programs,
including autoregressive decode to teacher-forcing conversion.
"""

import re
from typing import Optional, Tuple

# Check for optional MLIR Python bindings
try:
    from mlir import ir as mlir_ir
    HAVE_MLIR = True
except ImportError:
    mlir_ir = None
    HAVE_MLIR = False


def rewrite_decode_to_teacher_forcing(
    hlo_text: str,
    func_name: str = "main",
    teacher_arg_name: Optional[str] = None
) -> str:
    """
    Transform an autoregressive decode function to teacher-forcing variant.

    This transformation:
    1. Duplicates the specified function with a new name
    2. Adds a teacher_tokens argument
    3. Modifies the while loop to use teacher tokens instead of generated tokens

    Args:
        hlo_text: StableHLO module text
        func_name: Name of function to transform (default: "main")
        teacher_arg_name: Name for teacher tokens argument (default: auto-generated)

    Returns:
        Modified StableHLO text with new teacher-forcing function

    Raises:
        RuntimeError: If the function or while loop cannot be found
    """
    if HAVE_MLIR:
        try:
            return _rewrite_with_mlir(hlo_text, func_name, teacher_arg_name)
        except Exception:
            # Fall back to textual rewriting
            pass

    return _rewrite_textually(hlo_text, func_name, teacher_arg_name)


def _rewrite_textually(
    hlo_text: str,
    func_name: str = "main",
    teacher_arg_name: Optional[str] = None
) -> str:
    """Textual rewriting implementation."""

    # Find and extract the target function - handle various formats
    func_pattern = (
        r"(func\.func\s+(?:public\s+|private\s+)?@"
        + re.escape(func_name)
        + r"\s*\((?P<args>[^)]*)\)\s*->\s*\((?P<rets>[^)]*)\)\s*\{[^}]*?(?:\n\s*\}|\}\s*$))"
    )
    func_match = re.search(func_pattern, hlo_text, flags=re.S | re.M)

    if not func_match:
        raise RuntimeError(f"Could not find function @{func_name}")

    original_func = func_match.group(1)
    orig_args = func_match.group("args")
    orig_rets = func_match.group("rets")

    # Create new function signature
    new_func_name = func_name + "_teacher_forcing"
    arg_count = orig_args.count('%arg')
    new_arg = teacher_arg_name or f"%arg{arg_count}"
    new_args = orig_args + f", {new_arg}: tensor<?xi32>"

    # Create the new function
    new_func = original_func

    # Replace function signature
    has_public = "public" in original_func[:100]
    public_str = "public " if has_public else ""

    new_func = re.sub(
        r"func\.func (?:public )?@" + re.escape(func_name) + r"\(.*?\)\s*->\s*\(.*?\)",
        f"func.func {public_str}@{new_func_name}({new_args}) -> ({orig_rets})",
        new_func,
        count=1
    )

    # Find and modify the while loop's do block
    do_match = re.search(r"(\s+do\s*\{)(.*?)(\n\s{4}\})", new_func, flags=re.S)

    if not do_match:
        raise RuntimeError("Could not find while loop 'do' block")

    do_body = do_match.group(2)
    do_lines = do_body.split('\n')

    # Find the return statement
    ret_line_idx = None
    for i, line in enumerate(do_lines):
        if 'stablehlo.return' in line:
            ret_line_idx = i
            break

    if ret_line_idx is None:
        raise RuntimeError("Could not find return in do block")

    # Insert operations to get teacher token
    # Assuming the loop index is %iterArg_2 (third iteration argument)
    insert_ops = [
        f"      %tf_idx = tensor.from_elements %iterArg_2 : tensor<1xi32>",
        f"      %tf_slice = stablehlo.dynamic_slice {new_arg}, %tf_idx, slice_sizes = [1] : "
        f"(tensor<?xi32>, tensor<1xi32>) -> tensor<1xi32>",
        f"      %tf_token = stablehlo.reshape %tf_slice : (tensor<1xi32>) -> tensor<i32>"
    ]

    # Modify return statement to use teacher token
    ret_line = do_lines[ret_line_idx]
    ret_match = re.match(r"(\s*stablehlo\.return\s+)(.*?)(\s*:.*)", ret_line)

    if ret_match:
        indent = ret_match.group(1)
        operands = ret_match.group(2).split(',')
        types = ret_match.group(3)

        if len(operands) >= 4:
            # Replace the token operand (usually 4th in pattern: E, W, i, token, output)
            # Adjust index based on your specific while loop structure
            operands[3] = "%tf_token"
            new_ret = indent + ', '.join(operands) + types
            do_lines[ret_line_idx] = new_ret

    # Insert new ops before return
    do_lines = do_lines[:ret_line_idx] + insert_ops + do_lines[ret_line_idx:]

    # Reconstruct the do block
    new_do_body = '\n'.join(do_lines)
    new_func = (
        new_func[:do_match.start()] +
        do_match.group(1) + new_do_body + do_match.group(3) +
        new_func[do_match.end():]
    )

    # Append the new function to the module
    return hlo_text + "\n\n" + new_func + "\n"


def _rewrite_with_mlir(
    hlo_text: str,
    func_name: str = "main",
    teacher_arg_name: Optional[str] = None
) -> str:
    """MLIR-based rewriting implementation (when available)."""

    with mlir_ir.Context() as ctx, mlir_ir.Location.unknown():
        module = mlir_ir.Module.parse(hlo_text)

        # Find the target function
        func_ops = [
            op for op in module.body.operations
            if op.operation.name == "func.func"
        ]

        entry = None
        for op in func_ops:
            name = mlir_ir.StringAttr(op.attributes["sym_name"]).value
            if name == func_name:
                entry = op
                break

        if entry is None:
            raise RuntimeError(f"Could not find function @{func_name}")

        entry_name = mlir_ir.StringAttr(entry.attributes["sym_name"]).value

        # Build new function type with teacher tokens argument
        ftype = mlir_ir.FunctionType(entry.type)
        new_inputs = list(ftype.inputs)
        new_results = list(ftype.results)

        # Add dynamic 1D i32 tensor for teacher tokens
        i32 = mlir_ir.IntegerType.get_signless(32)
        dyn_1d_i32 = mlir_ir.RankedTensorType.get([-1], i32)
        new_inputs.append(dyn_1d_i32)

        new_ftype = mlir_ir.FunctionType.get(inputs=new_inputs, results=new_results)

        # Create new function
        new_name = entry_name + "_teacher_forcing"
        new_func = mlir_ir.Operation.create(
            "func.func",
            attributes={
                "sym_name": mlir_ir.StringAttr.get(new_name),
                "function_type": mlir_ir.TypeAttr.get(new_ftype),
                "sym_visibility": mlir_ir.StringAttr.get("public"),
            },
            regions=1,
        )

        new_func.operation.insert_after(entry.operation)

        # Set up new function body
        region = new_func.regions[0]
        block = mlir_ir.Block.create_at_start(region, new_inputs)
        teacher_tokens_value = block.arguments[-1]

        # Clone original function body
        old_block = entry.regions[0].blocks[0]
        value_map = mlir_ir.ValueMapping()
        for old_arg, new_arg in zip(old_block.arguments, block.arguments):
            value_map.map(old_arg, new_arg)

        for op in old_block.operations:
            cloned = op.operation.clone(value_map)
            block.append_operation(cloned)

        # Find and modify the while loop
        while_op = None
        for op in block.operations:
            if op.operation.name.endswith(".while"):
                while_op = op
                break

        if while_op is None:
            raise RuntimeError("Could not find while op")

        # Modify the while body
        body_region = while_op.regions[1]
        body_block = body_region.blocks[0]

        # Assume first block arg is loop index
        idx_val = body_block.arguments[0]

        # Insert teacher token extraction before terminator
        with mlir_ir.InsertionPoint(body_block.operations[-1]):
            # Create index tensor
            i1 = mlir_ir.Operation.create(
                "tensor.from_elements",
                operands=[idx_val],
                results=[mlir_ir.RankedTensorType.get([1], idx_val.type)],
            ).result

            # Dynamic slice to get teacher token
            slice_attr = mlir_ir.DenseI64ArrayAttr.get([1])
            dyn_slice = mlir_ir.Operation.create(
                "stablehlo.dynamic_slice",
                operands=[teacher_tokens_value, i1],
                results=[mlir_ir.RankedTensorType.get([1], idx_val.type)],
                attributes={"slice_sizes": slice_attr},
            ).result

            # Reshape to scalar
            next_id_scalar = mlir_ir.Operation.create(
                "stablehlo.reshape",
                operands=[dyn_slice],
                results=[idx_val.type],
            ).result

            # Patch the return to use teacher token
            ret = body_block.operations[-1]
            new_ret_operands = list(ret.operands)

            # Find the token position (typically index 3 or 4)
            # Adjust based on your specific loop structure
            if len(new_ret_operands) >= 4:
                new_ret_operands[3] = next_id_scalar

            new_ret = mlir_ir.Operation.create(
                ret.operation.name,
                operands=new_ret_operands,
            )
            ret.operation.replace_all_uses_with(new_ret.operation)
            ret.operation.erase()

        return str(module)


def extract_function(hlo_text: str, func_name: str = "main") -> str:
    """
    Extract a single function from a StableHLO module.

    Args:
        hlo_text: StableHLO module text
        func_name: Name of function to extract

    Returns:
        The extracted function text

    Raises:
        RuntimeError: If function not found
    """
    # More flexible pattern that handles different indentations
    pattern = (
        r"(func\.func\s+(?:public\s+|private\s+)?@"
        + re.escape(func_name)
        + r"\s*\([^)]*\)[^{]*\{[^}]*?(?:\n\s*\}|\}\s*$))"
    )
    match = re.search(pattern, hlo_text, flags=re.S | re.M)

    if not match:
        raise RuntimeError(f"Function @{func_name} not found")

    return match.group(1)


def list_functions(hlo_text: str) -> list[str]:
    """
    List all function names in a StableHLO module.

    Args:
        hlo_text: StableHLO module text

    Returns:
        List of function names
    """
    pattern = r"func\.func\s+(?:public\s+|private\s+)?@(\w+)"
    matches = re.findall(pattern, hlo_text)
    return matches