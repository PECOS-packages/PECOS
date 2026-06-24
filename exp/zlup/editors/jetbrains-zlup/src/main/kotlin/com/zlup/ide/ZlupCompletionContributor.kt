package com.zlup.ide

import com.intellij.codeInsight.completion.*
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.patterns.PlatformPatterns
import com.intellij.util.ProcessingContext
import com.intellij.icons.AllIcons

class ZlupCompletionContributor : CompletionContributor() {
    init {
        // Keywords
        extend(
            CompletionType.BASIC,
            PlatformPatterns.psiElement(),
            KeywordCompletionProvider()
        )
    }
}

class KeywordCompletionProvider : CompletionProvider<CompletionParameters>() {
    companion object {
        private val KEYWORDS = listOf(
            // Control flow
            "if", "else", "while", "for", "return", "break", "continue",
            // Declarations
            "fn", "pub", "const", "struct", "enum", "union", "error", "type",
            // Error handling
            "try", "catch", "orelse", "defer", "errdefer",
            // Modifiers
            "mut", "comptime", "inline", "packed", "extern",
            // Logical
            "and", "or", "not",
            // Literals
            "true", "false", "null", "undefined", "unit",
            // Other
            "in", "tick", "barrier"
        )

        private val TYPES = listOf(
            // Integer types
            "u1", "u8", "u16", "u32", "u64", "u128",
            "i8", "i16", "i32", "i64", "i128",
            "usize", "isize",
            // Float types
            "f16", "f32", "f64", "f128",
            // Angle type
            "a64",
            // Boolean
            "bool",
            // Quantum types
            "qubit", "bit", "qalloc",
            // Special
            "void", "type", "anytype", "anyerror", "anyfault"
        )

        private val GATES = listOf(
            // Single-qubit gates
            "h", "x", "y", "z", "s", "sdg", "t", "tdg",
            "sx", "sxdg", "sy", "sydg", "sz", "szdg",
            "f", "fdg", "f4", "f4dg",
            // Rotations
            "rx", "ry", "rz",
            // Two-qubit gates
            "cx", "cy", "cz", "ch", "swap", "iswap",
            "sxx", "sxxdg", "syy", "syydg", "szz", "szzdg", "rzz",
            // Three-qubit gates
            "ccx",
            // Measurement/preparation
            "mz", "mx", "my", "pz", "px", "py"
        )

        private val BUILTINS = listOf(
            "@import", "@sizeof", "@alignof", "@typeOf",
            "@intCast", "@floatCast", "@truncate",
            "@bitCast", "@ptrCast",
            "@min", "@max", "@clamp",
            "@sqrt", "@sin", "@cos", "@tan",
            "@log", "@log2", "@log10", "@exp",
            "@floor", "@ceil", "@round",
            "@abs", "@mod", "@divFloor", "@divTrunc"
        )

        private val STD_MODULES = listOf(
            "std.f64.pi", "std.f64.tau", "std.f64.e", "std.f64.sqrt2",
            "std.a64.quarter_turn", "std.a64.half_turn", "std.a64.t_angle",
            "std.math", "std.bits", "std.qec"
        )
    }

    override fun addCompletions(
        parameters: CompletionParameters,
        context: ProcessingContext,
        result: CompletionResultSet
    ) {
        // Add keywords
        for (keyword in KEYWORDS) {
            result.addElement(
                LookupElementBuilder.create(keyword)
                    .withIcon(AllIcons.Nodes.Favorite)
                    .withTypeText("keyword")
                    .bold()
            )
        }

        // Add types
        for (type in TYPES) {
            result.addElement(
                LookupElementBuilder.create(type)
                    .withIcon(AllIcons.Nodes.Class)
                    .withTypeText("type")
            )
        }

        // Add gates
        for (gate in GATES) {
            result.addElement(
                LookupElementBuilder.create(gate)
                    .withIcon(AllIcons.Nodes.Function)
                    .withTypeText("gate")
                    .withTailText(" (quantum gate)")
            )
        }

        // Add builtins
        for (builtin in BUILTINS) {
            result.addElement(
                LookupElementBuilder.create(builtin)
                    .withIcon(AllIcons.Nodes.Method)
                    .withTypeText("builtin")
            )
        }

        // Add common std library items
        for (module in STD_MODULES) {
            result.addElement(
                LookupElementBuilder.create(module)
                    .withIcon(AllIcons.Nodes.Module)
                    .withTypeText("std")
            )
        }

        // Add common code snippets
        result.addElement(
            LookupElementBuilder.create("fn main() -> unit {\n    \n}")
                .withPresentableText("fn main")
                .withIcon(AllIcons.Nodes.Function)
                .withTypeText("main function")
                .withInsertHandler { ctx, _ ->
                    ctx.editor.caretModel.moveToOffset(ctx.tailOffset - 2)
                }
        )

        result.addElement(
            LookupElementBuilder.create("q := qalloc()")
                .withPresentableText("qalloc")
                .withIcon(AllIcons.Nodes.Variable)
                .withTypeText("allocate qubits")
        )

        result.addElement(
            LookupElementBuilder.create("for i in 0..n {\n    \n}")
                .withPresentableText("for loop")
                .withIcon(AllIcons.Nodes.Favorite)
                .withTypeText("for loop")
        )

        result.addElement(
            LookupElementBuilder.create("if condition {\n    \n} else {\n    \n}")
                .withPresentableText("if-else")
                .withIcon(AllIcons.Nodes.Favorite)
                .withTypeText("if-else block")
        )
    }
}
