package com.zlup.ide

import com.intellij.lang.ASTNode
import com.intellij.lang.folding.FoldingBuilderEx
import com.intellij.lang.folding.FoldingDescriptor
import com.intellij.openapi.editor.Document
import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiElement
import com.intellij.psi.util.PsiTreeUtil

class ZlupFoldingBuilder : FoldingBuilderEx() {
    override fun buildFoldRegions(root: PsiElement, document: Document, quick: Boolean): Array<FoldingDescriptor> {
        val descriptors = mutableListOf<FoldingDescriptor>()
        val text = root.text

        // Find all brace pairs for folding
        findBracePairs(text, '{', '}', root, descriptors)

        // Find block comments
        findBlockComments(text, root, descriptors)

        return descriptors.toTypedArray()
    }

    private fun findBracePairs(
        text: String,
        openChar: Char,
        closeChar: Char,
        root: PsiElement,
        descriptors: MutableList<FoldingDescriptor>
    ) {
        val stack = mutableListOf<Int>()
        var i = 0

        while (i < text.length) {
            when {
                text[i] == openChar -> {
                    stack.add(i)
                }
                text[i] == closeChar && stack.isNotEmpty() -> {
                    val openIndex = stack.removeAt(stack.size - 1)
                    val closeIndex = i

                    // Only fold if the region spans multiple lines
                    val openLine = text.substring(0, openIndex).count { it == '\n' }
                    val closeLine = text.substring(0, closeIndex).count { it == '\n' }

                    if (closeLine > openLine) {
                        val range = TextRange(openIndex, closeIndex + 1)
                        if (range.length > 1) {
                            descriptors.add(FoldingDescriptor(root.node, range))
                        }
                    }
                }
                // Skip strings
                text[i] == '"' -> {
                    i++
                    while (i < text.length && text[i] != '"') {
                        if (text[i] == '\\' && i + 1 < text.length) i++
                        i++
                    }
                }
                // Skip line comments
                text[i] == '/' && i + 1 < text.length && text[i + 1] == '/' -> {
                    while (i < text.length && text[i] != '\n') i++
                }
                // Skip block comments
                text[i] == '/' && i + 1 < text.length && text[i + 1] == '*' -> {
                    i += 2
                    while (i + 1 < text.length && !(text[i] == '*' && text[i + 1] == '/')) i++
                    i++
                }
            }
            i++
        }
    }

    private fun findBlockComments(
        text: String,
        root: PsiElement,
        descriptors: MutableList<FoldingDescriptor>
    ) {
        var i = 0
        while (i < text.length - 1) {
            if (text[i] == '/' && text[i + 1] == '*') {
                val start = i
                i += 2
                while (i + 1 < text.length && !(text[i] == '*' && text[i + 1] == '/')) {
                    i++
                }
                val end = i + 2
                if (end <= text.length) {
                    val range = TextRange(start, end)
                    val openLine = text.substring(0, start).count { it == '\n' }
                    val closeLine = text.substring(0, end).count { it == '\n' }
                    if (closeLine > openLine && range.length > 2) {
                        descriptors.add(FoldingDescriptor(root.node, range))
                    }
                }
                i = end
            } else {
                i++
            }
        }
    }

    override fun getPlaceholderText(node: ASTNode): String {
        val text = node.text
        return when {
            text.startsWith("{") -> "{...}"
            text.startsWith("/*") -> "/*...*/"
            else -> "..."
        }
    }

    override fun isCollapsedByDefault(node: ASTNode): Boolean = false
}
