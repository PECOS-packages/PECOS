package com.zlup.ide

import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.fileTypes.SyntaxHighlighter
import com.intellij.openapi.options.colors.AttributesDescriptor
import com.intellij.openapi.options.colors.ColorDescriptor
import com.intellij.openapi.options.colors.ColorSettingsPage
import javax.swing.Icon

class ZlupColorSettingsPage : ColorSettingsPage {
    companion object {
        private val DESCRIPTORS = arrayOf(
            AttributesDescriptor("Keyword", ZlupSyntaxHighlighter.KEYWORD),
            AttributesDescriptor("Type", ZlupSyntaxHighlighter.TYPE),
            AttributesDescriptor("Gate", ZlupSyntaxHighlighter.GATE),
            AttributesDescriptor("Identifier", ZlupSyntaxHighlighter.IDENTIFIER),
            AttributesDescriptor("Number", ZlupSyntaxHighlighter.NUMBER),
            AttributesDescriptor("String", ZlupSyntaxHighlighter.STRING),
            AttributesDescriptor("Line Comment", ZlupSyntaxHighlighter.LINE_COMMENT),
            AttributesDescriptor("Block Comment", ZlupSyntaxHighlighter.BLOCK_COMMENT),
            AttributesDescriptor("Operator", ZlupSyntaxHighlighter.OPERATOR),
            AttributesDescriptor("Brackets", ZlupSyntaxHighlighter.BRACKETS),
            AttributesDescriptor("Braces", ZlupSyntaxHighlighter.BRACES),
            AttributesDescriptor("Parentheses", ZlupSyntaxHighlighter.PARENTHESES),
            AttributesDescriptor("Comma", ZlupSyntaxHighlighter.COMMA),
            AttributesDescriptor("Semicolon", ZlupSyntaxHighlighter.SEMICOLON),
            AttributesDescriptor("Dot", ZlupSyntaxHighlighter.DOT),
        )
    }

    override fun getIcon(): Icon = ZlupIcons.FILE

    override fun getHighlighter(): SyntaxHighlighter = ZlupSyntaxHighlighter()

    override fun getDemoText(): String = """
/// Bell state preparation
/// Creates entangled qubit pair
pub fn main() -> unit {
    // Allocate 2 qubits
    q := qalloc(2);

    /* Prepare and entangle */
    pz q;
    h q[0];
    cx (q[0], q[1]);

    // Measure results
    results: [2]u1 = mz([2]u1) [q[0], q[1]];

    if results[0] == 1 {
        x q[1];  // Apply correction
    }

    return unit;
}

const PI: f64 = 3.14159265358979;
const NUM_QUBITS: u32 = 4;
""".trimIndent()

    override fun getAdditionalHighlightingTagToDescriptorMap(): Map<String, TextAttributesKey>? = null

    override fun getAttributeDescriptors(): Array<AttributesDescriptor> = DESCRIPTORS

    override fun getColorDescriptors(): Array<ColorDescriptor> = ColorDescriptor.EMPTY_ARRAY

    override fun getDisplayName(): String = "Zlup"
}
