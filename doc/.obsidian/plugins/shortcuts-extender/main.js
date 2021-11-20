'use strict';

var obsidian = require('obsidian');

/*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global Reflect, Promise */

var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
    return extendStatics(d, b);
};

function __extends(d, b) {
    extendStatics(d, b);
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

function __awaiter(thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
}

function __generator(thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
}

var shortcutsExtender = /** @class */ (function (_super) {
    __extends(shortcutsExtender, _super);
    function shortcutsExtender() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    shortcutsExtender.prototype.onload = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            return __generator(this, function (_a) {
                console.log(this.app);
                this.addCommand({
                    id: "shortcut-exclamation-mark",
                    name: "Shortcut for ! symbol",
                    callback: function () { return _this.shortcutExclamationMark(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "1",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-At",
                    name: "Shortcut for @ symbol",
                    callback: function () { return _this.shortcutAt(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "2",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-hash",
                    name: "Shortcut for # symbol",
                    callback: function () { return _this.shortcutHash(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "3",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-dollar",
                    name: "Shortcut for $ symbol",
                    callback: function () { return _this.shortcutDollar(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "4",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-percent",
                    name: "Shortcut for % symbol",
                    callback: function () { return _this.shortcutPercent(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "5",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-circumflex",
                    name: "Shortcut for ^ symbol",
                    callback: function () { return _this.shortcutCircumflex(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "6",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-ampersand",
                    name: "Shortcut for & symbol",
                    callback: function () { return _this.shortcutAmpersand(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "7",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-less-than",
                    name: "Shortcut for < symbol",
                    callback: function () { return _this.shortcutLessThan(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "б",
                        },
                        {
                            modifiers: ["Alt"],
                            key: ",",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-greater-than",
                    name: "Shortcut for > symbol",
                    callback: function () { return _this.shortcutGreaterThan(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "ю",
                        },
                        {
                            modifiers: ["Alt"],
                            key: ".",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-left-square-bracket",
                    name: "Shortcut for [ symbol",
                    callback: function () { return _this.shortcutLeftSquareBracket(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "х",
                        },
                        {
                            modifiers: ["Alt"],
                            key: "[",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-right-square-bracket",
                    name: "Shortcut for ] symbol",
                    callback: function () { return _this.shortcutRightSquareBracket(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "ъ",
                        },
                        {
                            modifiers: ["Alt"],
                            key: "]",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-python-code",
                    name: "Shortcut for .py code fences",
                    callback: function () { return _this.shortcutPyCode(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt", "Shift"],
                            key: "ё",
                        },
                        {
                            modifiers: ["Alt", "Shift"],
                            key: "~",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-code",
                    name: "Shortcut for code fences (`)",
                    callback: function () { return _this.shortcutCodeFences(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "ё",
                        },
                        {
                            modifiers: ["Alt"],
                            key: "`",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-left-curly-bracket",
                    name: "Shortcut for { symbol",
                    callback: function () { return _this.shortcutLeftCurlyBracket(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt", "Shift"],
                            key: "х",
                        },
                        {
                            modifiers: ["Alt", "Shift"],
                            key: "{",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-right-curly-bracket",
                    name: "Shortcut for } symbol",
                    callback: function () { return _this.shortcutRightCurlyBracket(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt", "Shift"],
                            key: "ъ",
                        },
                        {
                            modifiers: ["Alt", "Shift"],
                            key: "}",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-Vertical-Line",
                    name: "Shortcut for | symbol",
                    callback: function () { return _this.shortcutVerticalLine(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt", "Shift"],
                            key: "|",
                        },
                        {
                            modifiers: ["Alt", "Shift"],
                            key: "\/",
                        },
                    ],
                });
                this.addCommand({
                    id: "shortcut-list-items",
                    name: "Creating list item from text",
                    callback: function () { return _this.shortcutListItems(); },
                    hotkeys: [
                        {
                            modifiers: ["Alt"],
                            key: "-",
                        },
                    ],
                });
                this.addCommand({
                    id: "heading-1",
                    name: "line into level 1 heading",
                    callback: function () { return _this.shortcutHeader1(); },
                    hotkeys: [
                        {
                            modifiers: ["Ctrl"],
                            key: "1",
                        },
                    ],
                });
                this.addCommand({
                    id: "heading-2",
                    name: "line into level 2 heading",
                    callback: function () { return _this.shortcutHeader2(); },
                    hotkeys: [
                        {
                            modifiers: ["Ctrl"],
                            key: "2",
                        },
                    ],
                });
                this.addCommand({
                    id: "heading-3",
                    name: "line into level 3 heading",
                    callback: function () { return _this.shortcutHeader3(); },
                    hotkeys: [
                        {
                            modifiers: ["Ctrl"],
                            key: "3",
                        },
                    ],
                });
                this.addCommand({
                    id: "heading-4",
                    name: "line into level 4 heading",
                    callback: function () { return _this.shortcutHeader4(); },
                    hotkeys: [
                        {
                            modifiers: ["Ctrl"],
                            key: "4",
                        },
                    ],
                });
                this.addCommand({
                    id: "heading-5",
                    name: "line into level 5 heading",
                    callback: function () { return _this.shortcutHeader5(); },
                    hotkeys: [
                        {
                            modifiers: ["Ctrl"],
                            key: "5",
                        },
                    ],
                });
                this.addCommand({
                    id: "heading-6",
                    name: "line into level 6 heading",
                    callback: function () { return _this.shortcutHeader6(); },
                    hotkeys: [
                        {
                            modifiers: ["Ctrl"],
                            key: "6",
                        },
                    ],
                });
                this.addCommand({
                    id: "heading-0",
                    name: "clearing of text formatting",
                    callback: function () { return _this.shortcutHeader0(); },
                    hotkeys: [
                        {
                            modifiers: ["Ctrl"],
                            key: "0",
                        },
                    ],
                });
                return [2 /*return*/];
            });
        });
    };
    shortcutsExtender.prototype.getSelectedText = function (editor) {
        //thanks to user "Argentina Ortega Sáinz" from the Obsidian community for this feature
        //For a long time I tried to do without such an approach, which resulted in several bugs and the impossibility of fixing them with non-crutches
        if (editor.somethingSelected()) {
            var cursorStart = editor.getCursor(true);
            var cursorEnd = editor.getCursor(false);
            var content = editor.getRange({ line: cursorStart.line, ch: 0 }, { line: cursorEnd.line, ch: editor.getLine(cursorEnd.line).length });
            return {
                start: { line: cursorStart.line, ch: 0 },
                end: {
                    line: cursorEnd.line,
                    ch: editor.getLine(cursorEnd.line).length,
                },
                content: content,
            };
        }
        else {
            // Toggle the todo in the line
            var lineNr = editor.getCursor().line;
            var contents = editor.getDoc().getLine(lineNr);
            var cursorStart = {
                line: lineNr,
                ch: 0,
            };
            var cursorEnd = {
                line: lineNr,
                ch: contents.length,
            };
            var content = editor.getRange(cursorStart, cursorEnd);
            return { start: cursorStart, end: cursorEnd, content: content };
        }
    };
    shortcutsExtender.prototype.shortcutExclamationMark = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("!");
    };
    shortcutsExtender.prototype.shortcutAt = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("@");
    };
    shortcutsExtender.prototype.shortcutHash = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("#");
    };
    shortcutsExtender.prototype.shortcutDollar = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("$");
    };
    shortcutsExtender.prototype.shortcutPercent = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("%");
    };
    shortcutsExtender.prototype.shortcutCircumflex = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("^");
    };
    shortcutsExtender.prototype.shortcutAmpersand = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("&");
    };
    shortcutsExtender.prototype.shortcutLessThan = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        if (selectedText) {
            editor.replaceSelection("<" + selectedText + ">");
        }
        else
            editor.replaceSelection("<");
    };
    shortcutsExtender.prototype.shortcutGreaterThan = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = this.getSelectedText(editor);
        var resultText = "> " + selectedText.content.split('\n').join("\n> ");
        editor.replaceRange(resultText, selectedText.start, selectedText.end);
    };
    shortcutsExtender.prototype.shortcutLeftSquareBracket = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        if (selectedText) {
            editor.replaceSelection("[[" + selectedText + "]]");
        }
        else
            editor.replaceSelection("[");
    };
    shortcutsExtender.prototype.shortcutRightSquareBracket = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("]");
    };
    shortcutsExtender.prototype.shortcutPyCode = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        if (selectedText) {
            editor.replaceSelection("```py\n" + selectedText + "\n```");
        }
        else
            editor.replaceSelection("`");
    };
    shortcutsExtender.prototype.shortcutCodeFences = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        if (selectedText) {
            editor.replaceSelection("`" + selectedText + "`");
        }
        else
            editor.replaceSelection("`");
    };
    shortcutsExtender.prototype.shortcutRightCurlyBracket = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("}");
    };
    shortcutsExtender.prototype.shortcutLeftCurlyBracket = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        if (selectedText) {
            editor.replaceSelection("{" + selectedText + "}");
        }
        else
            editor.replaceSelection("{");
    };
    shortcutsExtender.prototype.shortcutVerticalLine = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = editor.somethingSelected()
            ? editor.getSelection()
            : false;
        editor.replaceSelection("|");
    };
    shortcutsExtender.prototype.shortcutListItems = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = this.getSelectedText(editor);
        var resultText = "- " + selectedText.content.split('\n').join("\n- ");
        var resultTextCheck = resultText;
        while (resultTextCheck.search("- - ") >= 0) {
            resultTextCheck = resultTextCheck.replace("- - ", "	- ");
        }
        while (resultTextCheck.search("- 		") >= 0) {
            resultTextCheck = resultTextCheck.replace("- 		", "	- 	");
        }
        while (resultTextCheck.search("- 	- ") >= 0) {
            resultTextCheck = resultTextCheck.replace("- 	- ", "		- ");
        }
        editor.replaceRange(resultTextCheck, selectedText.start, selectedText.end);
    };
    shortcutsExtender.prototype.shortcutHeader0 = function () {
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = this.getSelectedText(editor);
        var j;
        var splittedSelectedText = selectedText.content.split("\n");
        for (j in splittedSelectedText) {
            while (splittedSelectedText[j].substring(0, 2) == "##") {
                splittedSelectedText[j] = splittedSelectedText[j].replace("##", "#");
            }
            while (splittedSelectedText[j].substring(0, 2) == "# ") {
                splittedSelectedText[j] = splittedSelectedText[j].substr(2);
            }
            while (splittedSelectedText[j].substring(0, 2) == "> ") {
                splittedSelectedText[j] = splittedSelectedText[j].substr(2);
            }
            while (splittedSelectedText[j].substring(0, 2) == "		") {
                splittedSelectedText[j] = splittedSelectedText[j].replace("		", "	");
            }
            while (splittedSelectedText[j].substring(0, 3) == "	- ") {
                splittedSelectedText[j] = splittedSelectedText[j].substr(3);
            }
            while (splittedSelectedText[j].substring(0, 2) == "- ") {
                splittedSelectedText[j] = splittedSelectedText[j].substr(2);
            }
            var re_digits = /^\d\.\s/;
            splittedSelectedText[j] = splittedSelectedText[j].replace(re_digits, "");
        }
        editor.replaceRange(splittedSelectedText, selectedText.start, selectedText.end);
    };
    shortcutsExtender.prototype.shortcutHeader1 = function () {
        this.shortcutHeader0();
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = this.getSelectedText(editor);
        //var clearText = selectedText.content.shortcutHeader0()
        var resultText = "# " + selectedText.content;
        editor.replaceRange(resultText, selectedText.start, selectedText.end);
    };
    shortcutsExtender.prototype.shortcutHeader2 = function () {
        this.shortcutHeader0();
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = this.getSelectedText(editor);
        var resultText = "## " + selectedText.content;
        editor.replaceRange(resultText, selectedText.start, selectedText.end);
    };
    shortcutsExtender.prototype.shortcutHeader3 = function () {
        this.shortcutHeader0();
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = this.getSelectedText(editor);
        var resultText = "### " + selectedText.content;
        editor.replaceRange(resultText, selectedText.start, selectedText.end);
    };
    shortcutsExtender.prototype.shortcutHeader4 = function () {
        this.shortcutHeader0();
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = this.getSelectedText(editor);
        var resultText = "#### " + selectedText.content;
        editor.replaceRange(resultText, selectedText.start, selectedText.end);
    };
    shortcutsExtender.prototype.shortcutHeader5 = function () {
        this.shortcutHeader0();
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = this.getSelectedText(editor);
        var resultText = "##### " + selectedText.content;
        editor.replaceRange(resultText, selectedText.start, selectedText.end);
    };
    shortcutsExtender.prototype.shortcutHeader6 = function () {
        this.shortcutHeader0();
        var activeLeaf = this.app.workspace.activeLeaf;
        var editor = activeLeaf.view.sourceMode.cmEditor;
        var selectedText = this.getSelectedText(editor);
        var resultText = "###### " + selectedText.content;
        editor.replaceRange(resultText, selectedText.start, selectedText.end);
    };
    return shortcutsExtender;
}(obsidian.Plugin));

module.exports = shortcutsExtender;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWFpbi5qcyIsInNvdXJjZXMiOlsibm9kZV9tb2R1bGVzL3RzbGliL3RzbGliLmVzNi5qcyIsIm1haW4udHMiXSwic291cmNlc0NvbnRlbnQiOlsiLyohICoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXHJcbkNvcHlyaWdodCAoYykgTWljcm9zb2Z0IENvcnBvcmF0aW9uLlxyXG5cclxuUGVybWlzc2lvbiB0byB1c2UsIGNvcHksIG1vZGlmeSwgYW5kL29yIGRpc3RyaWJ1dGUgdGhpcyBzb2Z0d2FyZSBmb3IgYW55XHJcbnB1cnBvc2Ugd2l0aCBvciB3aXRob3V0IGZlZSBpcyBoZXJlYnkgZ3JhbnRlZC5cclxuXHJcblRIRSBTT0ZUV0FSRSBJUyBQUk9WSURFRCBcIkFTIElTXCIgQU5EIFRIRSBBVVRIT1IgRElTQ0xBSU1TIEFMTCBXQVJSQU5USUVTIFdJVEhcclxuUkVHQVJEIFRPIFRISVMgU09GVFdBUkUgSU5DTFVESU5HIEFMTCBJTVBMSUVEIFdBUlJBTlRJRVMgT0YgTUVSQ0hBTlRBQklMSVRZXHJcbkFORCBGSVRORVNTLiBJTiBOTyBFVkVOVCBTSEFMTCBUSEUgQVVUSE9SIEJFIExJQUJMRSBGT1IgQU5ZIFNQRUNJQUwsIERJUkVDVCxcclxuSU5ESVJFQ1QsIE9SIENPTlNFUVVFTlRJQUwgREFNQUdFUyBPUiBBTlkgREFNQUdFUyBXSEFUU09FVkVSIFJFU1VMVElORyBGUk9NXHJcbkxPU1MgT0YgVVNFLCBEQVRBIE9SIFBST0ZJVFMsIFdIRVRIRVIgSU4gQU4gQUNUSU9OIE9GIENPTlRSQUNULCBORUdMSUdFTkNFIE9SXHJcbk9USEVSIFRPUlRJT1VTIEFDVElPTiwgQVJJU0lORyBPVVQgT0YgT1IgSU4gQ09OTkVDVElPTiBXSVRIIFRIRSBVU0UgT1JcclxuUEVSRk9STUFOQ0UgT0YgVEhJUyBTT0ZUV0FSRS5cclxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiogKi9cclxuLyogZ2xvYmFsIFJlZmxlY3QsIFByb21pc2UgKi9cclxuXHJcbnZhciBleHRlbmRTdGF0aWNzID0gZnVuY3Rpb24oZCwgYikge1xyXG4gICAgZXh0ZW5kU3RhdGljcyA9IE9iamVjdC5zZXRQcm90b3R5cGVPZiB8fFxyXG4gICAgICAgICh7IF9fcHJvdG9fXzogW10gfSBpbnN0YW5jZW9mIEFycmF5ICYmIGZ1bmN0aW9uIChkLCBiKSB7IGQuX19wcm90b19fID0gYjsgfSkgfHxcclxuICAgICAgICBmdW5jdGlvbiAoZCwgYikgeyBmb3IgKHZhciBwIGluIGIpIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwoYiwgcCkpIGRbcF0gPSBiW3BdOyB9O1xyXG4gICAgcmV0dXJuIGV4dGVuZFN0YXRpY3MoZCwgYik7XHJcbn07XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19leHRlbmRzKGQsIGIpIHtcclxuICAgIGV4dGVuZFN0YXRpY3MoZCwgYik7XHJcbiAgICBmdW5jdGlvbiBfXygpIHsgdGhpcy5jb25zdHJ1Y3RvciA9IGQ7IH1cclxuICAgIGQucHJvdG90eXBlID0gYiA9PT0gbnVsbCA/IE9iamVjdC5jcmVhdGUoYikgOiAoX18ucHJvdG90eXBlID0gYi5wcm90b3R5cGUsIG5ldyBfXygpKTtcclxufVxyXG5cclxuZXhwb3J0IHZhciBfX2Fzc2lnbiA9IGZ1bmN0aW9uKCkge1xyXG4gICAgX19hc3NpZ24gPSBPYmplY3QuYXNzaWduIHx8IGZ1bmN0aW9uIF9fYXNzaWduKHQpIHtcclxuICAgICAgICBmb3IgKHZhciBzLCBpID0gMSwgbiA9IGFyZ3VtZW50cy5sZW5ndGg7IGkgPCBuOyBpKyspIHtcclxuICAgICAgICAgICAgcyA9IGFyZ3VtZW50c1tpXTtcclxuICAgICAgICAgICAgZm9yICh2YXIgcCBpbiBzKSBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKHMsIHApKSB0W3BdID0gc1twXTtcclxuICAgICAgICB9XHJcbiAgICAgICAgcmV0dXJuIHQ7XHJcbiAgICB9XHJcbiAgICByZXR1cm4gX19hc3NpZ24uYXBwbHkodGhpcywgYXJndW1lbnRzKTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fcmVzdChzLCBlKSB7XHJcbiAgICB2YXIgdCA9IHt9O1xyXG4gICAgZm9yICh2YXIgcCBpbiBzKSBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKHMsIHApICYmIGUuaW5kZXhPZihwKSA8IDApXHJcbiAgICAgICAgdFtwXSA9IHNbcF07XHJcbiAgICBpZiAocyAhPSBudWxsICYmIHR5cGVvZiBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzID09PSBcImZ1bmN0aW9uXCIpXHJcbiAgICAgICAgZm9yICh2YXIgaSA9IDAsIHAgPSBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzKHMpOyBpIDwgcC5sZW5ndGg7IGkrKykge1xyXG4gICAgICAgICAgICBpZiAoZS5pbmRleE9mKHBbaV0pIDwgMCAmJiBPYmplY3QucHJvdG90eXBlLnByb3BlcnR5SXNFbnVtZXJhYmxlLmNhbGwocywgcFtpXSkpXHJcbiAgICAgICAgICAgICAgICB0W3BbaV1dID0gc1twW2ldXTtcclxuICAgICAgICB9XHJcbiAgICByZXR1cm4gdDtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fZGVjb3JhdGUoZGVjb3JhdG9ycywgdGFyZ2V0LCBrZXksIGRlc2MpIHtcclxuICAgIHZhciBjID0gYXJndW1lbnRzLmxlbmd0aCwgciA9IGMgPCAzID8gdGFyZ2V0IDogZGVzYyA9PT0gbnVsbCA/IGRlc2MgPSBPYmplY3QuZ2V0T3duUHJvcGVydHlEZXNjcmlwdG9yKHRhcmdldCwga2V5KSA6IGRlc2MsIGQ7XHJcbiAgICBpZiAodHlwZW9mIFJlZmxlY3QgPT09IFwib2JqZWN0XCIgJiYgdHlwZW9mIFJlZmxlY3QuZGVjb3JhdGUgPT09IFwiZnVuY3Rpb25cIikgciA9IFJlZmxlY3QuZGVjb3JhdGUoZGVjb3JhdG9ycywgdGFyZ2V0LCBrZXksIGRlc2MpO1xyXG4gICAgZWxzZSBmb3IgKHZhciBpID0gZGVjb3JhdG9ycy5sZW5ndGggLSAxOyBpID49IDA7IGktLSkgaWYgKGQgPSBkZWNvcmF0b3JzW2ldKSByID0gKGMgPCAzID8gZChyKSA6IGMgPiAzID8gZCh0YXJnZXQsIGtleSwgcikgOiBkKHRhcmdldCwga2V5KSkgfHwgcjtcclxuICAgIHJldHVybiBjID4gMyAmJiByICYmIE9iamVjdC5kZWZpbmVQcm9wZXJ0eSh0YXJnZXQsIGtleSwgciksIHI7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3BhcmFtKHBhcmFtSW5kZXgsIGRlY29yYXRvcikge1xyXG4gICAgcmV0dXJuIGZ1bmN0aW9uICh0YXJnZXQsIGtleSkgeyBkZWNvcmF0b3IodGFyZ2V0LCBrZXksIHBhcmFtSW5kZXgpOyB9XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX21ldGFkYXRhKG1ldGFkYXRhS2V5LCBtZXRhZGF0YVZhbHVlKSB7XHJcbiAgICBpZiAodHlwZW9mIFJlZmxlY3QgPT09IFwib2JqZWN0XCIgJiYgdHlwZW9mIFJlZmxlY3QubWV0YWRhdGEgPT09IFwiZnVuY3Rpb25cIikgcmV0dXJuIFJlZmxlY3QubWV0YWRhdGEobWV0YWRhdGFLZXksIG1ldGFkYXRhVmFsdWUpO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hd2FpdGVyKHRoaXNBcmcsIF9hcmd1bWVudHMsIFAsIGdlbmVyYXRvcikge1xyXG4gICAgZnVuY3Rpb24gYWRvcHQodmFsdWUpIHsgcmV0dXJuIHZhbHVlIGluc3RhbmNlb2YgUCA/IHZhbHVlIDogbmV3IFAoZnVuY3Rpb24gKHJlc29sdmUpIHsgcmVzb2x2ZSh2YWx1ZSk7IH0pOyB9XHJcbiAgICByZXR1cm4gbmV3IChQIHx8IChQID0gUHJvbWlzZSkpKGZ1bmN0aW9uIChyZXNvbHZlLCByZWplY3QpIHtcclxuICAgICAgICBmdW5jdGlvbiBmdWxmaWxsZWQodmFsdWUpIHsgdHJ5IHsgc3RlcChnZW5lcmF0b3IubmV4dCh2YWx1ZSkpOyB9IGNhdGNoIChlKSB7IHJlamVjdChlKTsgfSB9XHJcbiAgICAgICAgZnVuY3Rpb24gcmVqZWN0ZWQodmFsdWUpIHsgdHJ5IHsgc3RlcChnZW5lcmF0b3JbXCJ0aHJvd1wiXSh2YWx1ZSkpOyB9IGNhdGNoIChlKSB7IHJlamVjdChlKTsgfSB9XHJcbiAgICAgICAgZnVuY3Rpb24gc3RlcChyZXN1bHQpIHsgcmVzdWx0LmRvbmUgPyByZXNvbHZlKHJlc3VsdC52YWx1ZSkgOiBhZG9wdChyZXN1bHQudmFsdWUpLnRoZW4oZnVsZmlsbGVkLCByZWplY3RlZCk7IH1cclxuICAgICAgICBzdGVwKChnZW5lcmF0b3IgPSBnZW5lcmF0b3IuYXBwbHkodGhpc0FyZywgX2FyZ3VtZW50cyB8fCBbXSkpLm5leHQoKSk7XHJcbiAgICB9KTtcclxufVxyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fZ2VuZXJhdG9yKHRoaXNBcmcsIGJvZHkpIHtcclxuICAgIHZhciBfID0geyBsYWJlbDogMCwgc2VudDogZnVuY3Rpb24oKSB7IGlmICh0WzBdICYgMSkgdGhyb3cgdFsxXTsgcmV0dXJuIHRbMV07IH0sIHRyeXM6IFtdLCBvcHM6IFtdIH0sIGYsIHksIHQsIGc7XHJcbiAgICByZXR1cm4gZyA9IHsgbmV4dDogdmVyYigwKSwgXCJ0aHJvd1wiOiB2ZXJiKDEpLCBcInJldHVyblwiOiB2ZXJiKDIpIH0sIHR5cGVvZiBTeW1ib2wgPT09IFwiZnVuY3Rpb25cIiAmJiAoZ1tTeW1ib2wuaXRlcmF0b3JdID0gZnVuY3Rpb24oKSB7IHJldHVybiB0aGlzOyB9KSwgZztcclxuICAgIGZ1bmN0aW9uIHZlcmIobikgeyByZXR1cm4gZnVuY3Rpb24gKHYpIHsgcmV0dXJuIHN0ZXAoW24sIHZdKTsgfTsgfVxyXG4gICAgZnVuY3Rpb24gc3RlcChvcCkge1xyXG4gICAgICAgIGlmIChmKSB0aHJvdyBuZXcgVHlwZUVycm9yKFwiR2VuZXJhdG9yIGlzIGFscmVhZHkgZXhlY3V0aW5nLlwiKTtcclxuICAgICAgICB3aGlsZSAoXykgdHJ5IHtcclxuICAgICAgICAgICAgaWYgKGYgPSAxLCB5ICYmICh0ID0gb3BbMF0gJiAyID8geVtcInJldHVyblwiXSA6IG9wWzBdID8geVtcInRocm93XCJdIHx8ICgodCA9IHlbXCJyZXR1cm5cIl0pICYmIHQuY2FsbCh5KSwgMCkgOiB5Lm5leHQpICYmICEodCA9IHQuY2FsbCh5LCBvcFsxXSkpLmRvbmUpIHJldHVybiB0O1xyXG4gICAgICAgICAgICBpZiAoeSA9IDAsIHQpIG9wID0gW29wWzBdICYgMiwgdC52YWx1ZV07XHJcbiAgICAgICAgICAgIHN3aXRjaCAob3BbMF0pIHtcclxuICAgICAgICAgICAgICAgIGNhc2UgMDogY2FzZSAxOiB0ID0gb3A7IGJyZWFrO1xyXG4gICAgICAgICAgICAgICAgY2FzZSA0OiBfLmxhYmVsKys7IHJldHVybiB7IHZhbHVlOiBvcFsxXSwgZG9uZTogZmFsc2UgfTtcclxuICAgICAgICAgICAgICAgIGNhc2UgNTogXy5sYWJlbCsrOyB5ID0gb3BbMV07IG9wID0gWzBdOyBjb250aW51ZTtcclxuICAgICAgICAgICAgICAgIGNhc2UgNzogb3AgPSBfLm9wcy5wb3AoKTsgXy50cnlzLnBvcCgpOyBjb250aW51ZTtcclxuICAgICAgICAgICAgICAgIGRlZmF1bHQ6XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKCEodCA9IF8udHJ5cywgdCA9IHQubGVuZ3RoID4gMCAmJiB0W3QubGVuZ3RoIC0gMV0pICYmIChvcFswXSA9PT0gNiB8fCBvcFswXSA9PT0gMikpIHsgXyA9IDA7IGNvbnRpbnVlOyB9XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKG9wWzBdID09PSAzICYmICghdCB8fCAob3BbMV0gPiB0WzBdICYmIG9wWzFdIDwgdFszXSkpKSB7IF8ubGFiZWwgPSBvcFsxXTsgYnJlYWs7IH1cclxuICAgICAgICAgICAgICAgICAgICBpZiAob3BbMF0gPT09IDYgJiYgXy5sYWJlbCA8IHRbMV0pIHsgXy5sYWJlbCA9IHRbMV07IHQgPSBvcDsgYnJlYWs7IH1cclxuICAgICAgICAgICAgICAgICAgICBpZiAodCAmJiBfLmxhYmVsIDwgdFsyXSkgeyBfLmxhYmVsID0gdFsyXTsgXy5vcHMucHVzaChvcCk7IGJyZWFrOyB9XHJcbiAgICAgICAgICAgICAgICAgICAgaWYgKHRbMl0pIF8ub3BzLnBvcCgpO1xyXG4gICAgICAgICAgICAgICAgICAgIF8udHJ5cy5wb3AoKTsgY29udGludWU7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgb3AgPSBib2R5LmNhbGwodGhpc0FyZywgXyk7XHJcbiAgICAgICAgfSBjYXRjaCAoZSkgeyBvcCA9IFs2LCBlXTsgeSA9IDA7IH0gZmluYWxseSB7IGYgPSB0ID0gMDsgfVxyXG4gICAgICAgIGlmIChvcFswXSAmIDUpIHRocm93IG9wWzFdOyByZXR1cm4geyB2YWx1ZTogb3BbMF0gPyBvcFsxXSA6IHZvaWQgMCwgZG9uZTogdHJ1ZSB9O1xyXG4gICAgfVxyXG59XHJcblxyXG5leHBvcnQgdmFyIF9fY3JlYXRlQmluZGluZyA9IE9iamVjdC5jcmVhdGUgPyAoZnVuY3Rpb24obywgbSwgaywgazIpIHtcclxuICAgIGlmIChrMiA9PT0gdW5kZWZpbmVkKSBrMiA9IGs7XHJcbiAgICBPYmplY3QuZGVmaW5lUHJvcGVydHkobywgazIsIHsgZW51bWVyYWJsZTogdHJ1ZSwgZ2V0OiBmdW5jdGlvbigpIHsgcmV0dXJuIG1ba107IH0gfSk7XHJcbn0pIDogKGZ1bmN0aW9uKG8sIG0sIGssIGsyKSB7XHJcbiAgICBpZiAoazIgPT09IHVuZGVmaW5lZCkgazIgPSBrO1xyXG4gICAgb1trMl0gPSBtW2tdO1xyXG59KTtcclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2V4cG9ydFN0YXIobSwgbykge1xyXG4gICAgZm9yICh2YXIgcCBpbiBtKSBpZiAocCAhPT0gXCJkZWZhdWx0XCIgJiYgIU9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvLCBwKSkgX19jcmVhdGVCaW5kaW5nKG8sIG0sIHApO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX192YWx1ZXMobykge1xyXG4gICAgdmFyIHMgPSB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgU3ltYm9sLml0ZXJhdG9yLCBtID0gcyAmJiBvW3NdLCBpID0gMDtcclxuICAgIGlmIChtKSByZXR1cm4gbS5jYWxsKG8pO1xyXG4gICAgaWYgKG8gJiYgdHlwZW9mIG8ubGVuZ3RoID09PSBcIm51bWJlclwiKSByZXR1cm4ge1xyXG4gICAgICAgIG5leHQ6IGZ1bmN0aW9uICgpIHtcclxuICAgICAgICAgICAgaWYgKG8gJiYgaSA+PSBvLmxlbmd0aCkgbyA9IHZvaWQgMDtcclxuICAgICAgICAgICAgcmV0dXJuIHsgdmFsdWU6IG8gJiYgb1tpKytdLCBkb25lOiAhbyB9O1xyXG4gICAgICAgIH1cclxuICAgIH07XHJcbiAgICB0aHJvdyBuZXcgVHlwZUVycm9yKHMgPyBcIk9iamVjdCBpcyBub3QgaXRlcmFibGUuXCIgOiBcIlN5bWJvbC5pdGVyYXRvciBpcyBub3QgZGVmaW5lZC5cIik7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3JlYWQobywgbikge1xyXG4gICAgdmFyIG0gPSB0eXBlb2YgU3ltYm9sID09PSBcImZ1bmN0aW9uXCIgJiYgb1tTeW1ib2wuaXRlcmF0b3JdO1xyXG4gICAgaWYgKCFtKSByZXR1cm4gbztcclxuICAgIHZhciBpID0gbS5jYWxsKG8pLCByLCBhciA9IFtdLCBlO1xyXG4gICAgdHJ5IHtcclxuICAgICAgICB3aGlsZSAoKG4gPT09IHZvaWQgMCB8fCBuLS0gPiAwKSAmJiAhKHIgPSBpLm5leHQoKSkuZG9uZSkgYXIucHVzaChyLnZhbHVlKTtcclxuICAgIH1cclxuICAgIGNhdGNoIChlcnJvcikgeyBlID0geyBlcnJvcjogZXJyb3IgfTsgfVxyXG4gICAgZmluYWxseSB7XHJcbiAgICAgICAgdHJ5IHtcclxuICAgICAgICAgICAgaWYgKHIgJiYgIXIuZG9uZSAmJiAobSA9IGlbXCJyZXR1cm5cIl0pKSBtLmNhbGwoaSk7XHJcbiAgICAgICAgfVxyXG4gICAgICAgIGZpbmFsbHkgeyBpZiAoZSkgdGhyb3cgZS5lcnJvcjsgfVxyXG4gICAgfVxyXG4gICAgcmV0dXJuIGFyO1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19zcHJlYWQoKSB7XHJcbiAgICBmb3IgKHZhciBhciA9IFtdLCBpID0gMDsgaSA8IGFyZ3VtZW50cy5sZW5ndGg7IGkrKylcclxuICAgICAgICBhciA9IGFyLmNvbmNhdChfX3JlYWQoYXJndW1lbnRzW2ldKSk7XHJcbiAgICByZXR1cm4gYXI7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX3NwcmVhZEFycmF5cygpIHtcclxuICAgIGZvciAodmFyIHMgPSAwLCBpID0gMCwgaWwgPSBhcmd1bWVudHMubGVuZ3RoOyBpIDwgaWw7IGkrKykgcyArPSBhcmd1bWVudHNbaV0ubGVuZ3RoO1xyXG4gICAgZm9yICh2YXIgciA9IEFycmF5KHMpLCBrID0gMCwgaSA9IDA7IGkgPCBpbDsgaSsrKVxyXG4gICAgICAgIGZvciAodmFyIGEgPSBhcmd1bWVudHNbaV0sIGogPSAwLCBqbCA9IGEubGVuZ3RoOyBqIDwgamw7IGorKywgaysrKVxyXG4gICAgICAgICAgICByW2tdID0gYVtqXTtcclxuICAgIHJldHVybiByO1xyXG59O1xyXG5cclxuZXhwb3J0IGZ1bmN0aW9uIF9fYXdhaXQodikge1xyXG4gICAgcmV0dXJuIHRoaXMgaW5zdGFuY2VvZiBfX2F3YWl0ID8gKHRoaXMudiA9IHYsIHRoaXMpIDogbmV3IF9fYXdhaXQodik7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2FzeW5jR2VuZXJhdG9yKHRoaXNBcmcsIF9hcmd1bWVudHMsIGdlbmVyYXRvcikge1xyXG4gICAgaWYgKCFTeW1ib2wuYXN5bmNJdGVyYXRvcikgdGhyb3cgbmV3IFR5cGVFcnJvcihcIlN5bWJvbC5hc3luY0l0ZXJhdG9yIGlzIG5vdCBkZWZpbmVkLlwiKTtcclxuICAgIHZhciBnID0gZ2VuZXJhdG9yLmFwcGx5KHRoaXNBcmcsIF9hcmd1bWVudHMgfHwgW10pLCBpLCBxID0gW107XHJcbiAgICByZXR1cm4gaSA9IHt9LCB2ZXJiKFwibmV4dFwiKSwgdmVyYihcInRocm93XCIpLCB2ZXJiKFwicmV0dXJuXCIpLCBpW1N5bWJvbC5hc3luY0l0ZXJhdG9yXSA9IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHRoaXM7IH0sIGk7XHJcbiAgICBmdW5jdGlvbiB2ZXJiKG4pIHsgaWYgKGdbbl0pIGlbbl0gPSBmdW5jdGlvbiAodikgeyByZXR1cm4gbmV3IFByb21pc2UoZnVuY3Rpb24gKGEsIGIpIHsgcS5wdXNoKFtuLCB2LCBhLCBiXSkgPiAxIHx8IHJlc3VtZShuLCB2KTsgfSk7IH07IH1cclxuICAgIGZ1bmN0aW9uIHJlc3VtZShuLCB2KSB7IHRyeSB7IHN0ZXAoZ1tuXSh2KSk7IH0gY2F0Y2ggKGUpIHsgc2V0dGxlKHFbMF1bM10sIGUpOyB9IH1cclxuICAgIGZ1bmN0aW9uIHN0ZXAocikgeyByLnZhbHVlIGluc3RhbmNlb2YgX19hd2FpdCA/IFByb21pc2UucmVzb2x2ZShyLnZhbHVlLnYpLnRoZW4oZnVsZmlsbCwgcmVqZWN0KSA6IHNldHRsZShxWzBdWzJdLCByKTsgfVxyXG4gICAgZnVuY3Rpb24gZnVsZmlsbCh2YWx1ZSkgeyByZXN1bWUoXCJuZXh0XCIsIHZhbHVlKTsgfVxyXG4gICAgZnVuY3Rpb24gcmVqZWN0KHZhbHVlKSB7IHJlc3VtZShcInRocm93XCIsIHZhbHVlKTsgfVxyXG4gICAgZnVuY3Rpb24gc2V0dGxlKGYsIHYpIHsgaWYgKGYodiksIHEuc2hpZnQoKSwgcS5sZW5ndGgpIHJlc3VtZShxWzBdWzBdLCBxWzBdWzFdKTsgfVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19hc3luY0RlbGVnYXRvcihvKSB7XHJcbiAgICB2YXIgaSwgcDtcclxuICAgIHJldHVybiBpID0ge30sIHZlcmIoXCJuZXh0XCIpLCB2ZXJiKFwidGhyb3dcIiwgZnVuY3Rpb24gKGUpIHsgdGhyb3cgZTsgfSksIHZlcmIoXCJyZXR1cm5cIiksIGlbU3ltYm9sLml0ZXJhdG9yXSA9IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHRoaXM7IH0sIGk7XHJcbiAgICBmdW5jdGlvbiB2ZXJiKG4sIGYpIHsgaVtuXSA9IG9bbl0gPyBmdW5jdGlvbiAodikgeyByZXR1cm4gKHAgPSAhcCkgPyB7IHZhbHVlOiBfX2F3YWl0KG9bbl0odikpLCBkb25lOiBuID09PSBcInJldHVyblwiIH0gOiBmID8gZih2KSA6IHY7IH0gOiBmOyB9XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2FzeW5jVmFsdWVzKG8pIHtcclxuICAgIGlmICghU3ltYm9sLmFzeW5jSXRlcmF0b3IpIHRocm93IG5ldyBUeXBlRXJyb3IoXCJTeW1ib2wuYXN5bmNJdGVyYXRvciBpcyBub3QgZGVmaW5lZC5cIik7XHJcbiAgICB2YXIgbSA9IG9bU3ltYm9sLmFzeW5jSXRlcmF0b3JdLCBpO1xyXG4gICAgcmV0dXJuIG0gPyBtLmNhbGwobykgOiAobyA9IHR5cGVvZiBfX3ZhbHVlcyA9PT0gXCJmdW5jdGlvblwiID8gX192YWx1ZXMobykgOiBvW1N5bWJvbC5pdGVyYXRvcl0oKSwgaSA9IHt9LCB2ZXJiKFwibmV4dFwiKSwgdmVyYihcInRocm93XCIpLCB2ZXJiKFwicmV0dXJuXCIpLCBpW1N5bWJvbC5hc3luY0l0ZXJhdG9yXSA9IGZ1bmN0aW9uICgpIHsgcmV0dXJuIHRoaXM7IH0sIGkpO1xyXG4gICAgZnVuY3Rpb24gdmVyYihuKSB7IGlbbl0gPSBvW25dICYmIGZ1bmN0aW9uICh2KSB7IHJldHVybiBuZXcgUHJvbWlzZShmdW5jdGlvbiAocmVzb2x2ZSwgcmVqZWN0KSB7IHYgPSBvW25dKHYpLCBzZXR0bGUocmVzb2x2ZSwgcmVqZWN0LCB2LmRvbmUsIHYudmFsdWUpOyB9KTsgfTsgfVxyXG4gICAgZnVuY3Rpb24gc2V0dGxlKHJlc29sdmUsIHJlamVjdCwgZCwgdikgeyBQcm9taXNlLnJlc29sdmUodikudGhlbihmdW5jdGlvbih2KSB7IHJlc29sdmUoeyB2YWx1ZTogdiwgZG9uZTogZCB9KTsgfSwgcmVqZWN0KTsgfVxyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19tYWtlVGVtcGxhdGVPYmplY3QoY29va2VkLCByYXcpIHtcclxuICAgIGlmIChPYmplY3QuZGVmaW5lUHJvcGVydHkpIHsgT2JqZWN0LmRlZmluZVByb3BlcnR5KGNvb2tlZCwgXCJyYXdcIiwgeyB2YWx1ZTogcmF3IH0pOyB9IGVsc2UgeyBjb29rZWQucmF3ID0gcmF3OyB9XHJcbiAgICByZXR1cm4gY29va2VkO1xyXG59O1xyXG5cclxudmFyIF9fc2V0TW9kdWxlRGVmYXVsdCA9IE9iamVjdC5jcmVhdGUgPyAoZnVuY3Rpb24obywgdikge1xyXG4gICAgT2JqZWN0LmRlZmluZVByb3BlcnR5KG8sIFwiZGVmYXVsdFwiLCB7IGVudW1lcmFibGU6IHRydWUsIHZhbHVlOiB2IH0pO1xyXG59KSA6IGZ1bmN0aW9uKG8sIHYpIHtcclxuICAgIG9bXCJkZWZhdWx0XCJdID0gdjtcclxufTtcclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2ltcG9ydFN0YXIobW9kKSB7XHJcbiAgICBpZiAobW9kICYmIG1vZC5fX2VzTW9kdWxlKSByZXR1cm4gbW9kO1xyXG4gICAgdmFyIHJlc3VsdCA9IHt9O1xyXG4gICAgaWYgKG1vZCAhPSBudWxsKSBmb3IgKHZhciBrIGluIG1vZCkgaWYgKGsgIT09IFwiZGVmYXVsdFwiICYmIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChtb2QsIGspKSBfX2NyZWF0ZUJpbmRpbmcocmVzdWx0LCBtb2QsIGspO1xyXG4gICAgX19zZXRNb2R1bGVEZWZhdWx0KHJlc3VsdCwgbW9kKTtcclxuICAgIHJldHVybiByZXN1bHQ7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2ltcG9ydERlZmF1bHQobW9kKSB7XHJcbiAgICByZXR1cm4gKG1vZCAmJiBtb2QuX19lc01vZHVsZSkgPyBtb2QgOiB7IGRlZmF1bHQ6IG1vZCB9O1xyXG59XHJcblxyXG5leHBvcnQgZnVuY3Rpb24gX19jbGFzc1ByaXZhdGVGaWVsZEdldChyZWNlaXZlciwgcHJpdmF0ZU1hcCkge1xyXG4gICAgaWYgKCFwcml2YXRlTWFwLmhhcyhyZWNlaXZlcikpIHtcclxuICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKFwiYXR0ZW1wdGVkIHRvIGdldCBwcml2YXRlIGZpZWxkIG9uIG5vbi1pbnN0YW5jZVwiKTtcclxuICAgIH1cclxuICAgIHJldHVybiBwcml2YXRlTWFwLmdldChyZWNlaXZlcik7XHJcbn1cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBfX2NsYXNzUHJpdmF0ZUZpZWxkU2V0KHJlY2VpdmVyLCBwcml2YXRlTWFwLCB2YWx1ZSkge1xyXG4gICAgaWYgKCFwcml2YXRlTWFwLmhhcyhyZWNlaXZlcikpIHtcclxuICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKFwiYXR0ZW1wdGVkIHRvIHNldCBwcml2YXRlIGZpZWxkIG9uIG5vbi1pbnN0YW5jZVwiKTtcclxuICAgIH1cclxuICAgIHByaXZhdGVNYXAuc2V0KHJlY2VpdmVyLCB2YWx1ZSk7XHJcbiAgICByZXR1cm4gdmFsdWU7XHJcbn1cclxuIiwiaW1wb3J0IHsgUGx1Z2luIH0gZnJvbSBcIm9ic2lkaWFuXCI7XG5cbmV4cG9ydCBkZWZhdWx0IGNsYXNzIHNob3J0Y3V0c0V4dGVuZGVyIGV4dGVuZHMgUGx1Z2luIHtcbiAgYXN5bmMgb25sb2FkKCkge1xuICAgIGNvbnNvbGUubG9nKHRoaXMuYXBwKTtcblxuICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICBpZDogXCJzaG9ydGN1dC1leGNsYW1hdGlvbi1tYXJrXCIsXG4gICAgICBuYW1lOiBcIlNob3J0Y3V0IGZvciAhIHN5bWJvbFwiLFxuICAgICAgY2FsbGJhY2s6ICgpID0+IHRoaXMuc2hvcnRjdXRFeGNsYW1hdGlvbk1hcmsoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCJdLFxuICAgICAgICAgIGtleTogXCIxXCIsXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgIGlkOiBcInNob3J0Y3V0LUF0XCIsXG4gICAgICBuYW1lOiBcIlNob3J0Y3V0IGZvciBAIHN5bWJvbFwiLFxuICAgICAgY2FsbGJhY2s6ICgpID0+IHRoaXMuc2hvcnRjdXRBdCgpLFxuICAgICAgaG90a2V5czogW1xuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJBbHRcIl0sXG4gICAgICAgICAga2V5OiBcIjJcIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwic2hvcnRjdXQtaGFzaFwiLFxuICAgICAgbmFtZTogXCJTaG9ydGN1dCBmb3IgIyBzeW1ib2xcIixcbiAgICAgIGNhbGxiYWNrOiAoKSA9PiB0aGlzLnNob3J0Y3V0SGFzaCgpLFxuICAgICAgaG90a2V5czogW1xuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJBbHRcIl0sXG4gICAgICAgICAga2V5OiBcIjNcIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwic2hvcnRjdXQtZG9sbGFyXCIsXG4gICAgICBuYW1lOiBcIlNob3J0Y3V0IGZvciAkIHN5bWJvbFwiLFxuICAgICAgY2FsbGJhY2s6ICgpID0+IHRoaXMuc2hvcnRjdXREb2xsYXIoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCJdLFxuICAgICAgICAgIGtleTogXCI0XCIsXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgIGlkOiBcInNob3J0Y3V0LXBlcmNlbnRcIixcbiAgICAgIG5hbWU6IFwiU2hvcnRjdXQgZm9yICUgc3ltYm9sXCIsXG4gICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5zaG9ydGN1dFBlcmNlbnQoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCJdLFxuICAgICAgICAgIGtleTogXCI1XCIsXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgIGlkOiBcInNob3J0Y3V0LWNpcmN1bWZsZXhcIixcbiAgICAgIG5hbWU6IFwiU2hvcnRjdXQgZm9yIF4gc3ltYm9sXCIsXG4gICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5zaG9ydGN1dENpcmN1bWZsZXgoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCJdLFxuICAgICAgICAgIGtleTogXCI2XCIsXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgIGlkOiBcInNob3J0Y3V0LWFtcGVyc2FuZFwiLFxuICAgICAgbmFtZTogXCJTaG9ydGN1dCBmb3IgJiBzeW1ib2xcIixcbiAgICAgIGNhbGxiYWNrOiAoKSA9PiB0aGlzLnNob3J0Y3V0QW1wZXJzYW5kKCksXG4gICAgICBob3RrZXlzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBtb2RpZmllcnM6IFtcIkFsdFwiXSxcbiAgICAgICAgICBrZXk6IFwiN1wiLFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICBpZDogXCJzaG9ydGN1dC1sZXNzLXRoYW5cIixcbiAgICAgIG5hbWU6IFwiU2hvcnRjdXQgZm9yIDwgc3ltYm9sXCIsXG4gICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5zaG9ydGN1dExlc3NUaGFuKCksXG4gICAgICBob3RrZXlzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBtb2RpZmllcnM6IFtcIkFsdFwiXSxcbiAgICAgICAgICBrZXk6IFwi0LFcIixcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCJdLFxuICAgICAgICAgIGtleTogXCIsXCIsXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgIGlkOiBcInNob3J0Y3V0LWdyZWF0ZXItdGhhblwiLFxuICAgICAgbmFtZTogXCJTaG9ydGN1dCBmb3IgPiBzeW1ib2xcIixcbiAgICAgIGNhbGxiYWNrOiAoKSA9PiB0aGlzLnNob3J0Y3V0R3JlYXRlclRoYW4oKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCJdLFxuICAgICAgICAgIGtleTogXCLRjlwiLFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJBbHRcIl0sXG4gICAgICAgICAga2V5OiBcIi5cIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwic2hvcnRjdXQtbGVmdC1zcXVhcmUtYnJhY2tldFwiLFxuICAgICAgbmFtZTogXCJTaG9ydGN1dCBmb3IgWyBzeW1ib2xcIixcbiAgICAgIGNhbGxiYWNrOiAoKSA9PiB0aGlzLnNob3J0Y3V0TGVmdFNxdWFyZUJyYWNrZXQoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCJdLFxuICAgICAgICAgIGtleTogXCLRhVwiLFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJBbHRcIl0sXG4gICAgICAgICAga2V5OiBcIltcIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwic2hvcnRjdXQtcmlnaHQtc3F1YXJlLWJyYWNrZXRcIixcbiAgICAgIG5hbWU6IFwiU2hvcnRjdXQgZm9yIF0gc3ltYm9sXCIsXG4gICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5zaG9ydGN1dFJpZ2h0U3F1YXJlQnJhY2tldCgpLFxuICAgICAgaG90a2V5czogW1xuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJBbHRcIl0sXG4gICAgICAgICAga2V5OiBcItGKXCIsXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBtb2RpZmllcnM6IFtcIkFsdFwiXSxcbiAgICAgICAgICBrZXk6IFwiXVwiLFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICBpZDogXCJzaG9ydGN1dC1weXRob24tY29kZVwiLFxuICAgICAgbmFtZTogXCJTaG9ydGN1dCBmb3IgLnB5IGNvZGUgZmVuY2VzXCIsXG4gICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5zaG9ydGN1dFB5Q29kZSgpLFxuICAgICAgaG90a2V5czogW1xuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJBbHRcIiwgXCJTaGlmdFwiXSxcbiAgICAgICAgICBrZXk6IFwi0ZFcIixcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCIsIFwiU2hpZnRcIl0sXG4gICAgICAgICAga2V5OiBcIn5cIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwic2hvcnRjdXQtY29kZVwiLFxuICAgICAgbmFtZTogXCJTaG9ydGN1dCBmb3IgY29kZSBmZW5jZXMgKGApXCIsXG4gICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5zaG9ydGN1dENvZGVGZW5jZXMoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCJdLFxuICAgICAgICAgIGtleTogXCLRkVwiLFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJBbHRcIl0sXG4gICAgICAgICAga2V5OiBcImBcIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwic2hvcnRjdXQtbGVmdC1jdXJseS1icmFja2V0XCIsXG4gICAgICBuYW1lOiBcIlNob3J0Y3V0IGZvciB7IHN5bWJvbFwiLFxuICAgICAgY2FsbGJhY2s6ICgpID0+IHRoaXMuc2hvcnRjdXRMZWZ0Q3VybHlCcmFja2V0KCksXG4gICAgICBob3RrZXlzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBtb2RpZmllcnM6IFtcIkFsdFwiLCBcIlNoaWZ0XCJdLFxuICAgICAgICAgIGtleTogXCLRhVwiLFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJBbHRcIiwgXCJTaGlmdFwiXSxcbiAgICAgICAgICBrZXk6IFwie1wiLFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICBpZDogXCJzaG9ydGN1dC1yaWdodC1jdXJseS1icmFja2V0XCIsXG4gICAgICBuYW1lOiBcIlNob3J0Y3V0IGZvciB9IHN5bWJvbFwiLFxuICAgICAgY2FsbGJhY2s6ICgpID0+IHRoaXMuc2hvcnRjdXRSaWdodEN1cmx5QnJhY2tldCgpLFxuICAgICAgaG90a2V5czogW1xuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJBbHRcIiwgXCJTaGlmdFwiXSxcbiAgICAgICAgICBrZXk6IFwi0YpcIixcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCIsIFwiU2hpZnRcIl0sXG4gICAgICAgICAga2V5OiBcIn1cIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwic2hvcnRjdXQtVmVydGljYWwtTGluZVwiLFxuICAgICAgbmFtZTogXCJTaG9ydGN1dCBmb3IgfCBzeW1ib2xcIixcbiAgICAgIGNhbGxiYWNrOiAoKSA9PiB0aGlzLnNob3J0Y3V0VmVydGljYWxMaW5lKCksXG4gICAgICBob3RrZXlzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBtb2RpZmllcnM6IFtcIkFsdFwiLCBcIlNoaWZ0XCJdLFxuICAgICAgICAgIGtleTogXCJ8XCIsXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBtb2RpZmllcnM6IFtcIkFsdFwiLCBcIlNoaWZ0XCJdLFxuICAgICAgICAgIGtleTogXCJcXC9cIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgIGlkOiBcInNob3J0Y3V0LWxpc3QtaXRlbXNcIixcbiAgICAgIG5hbWU6IFwiQ3JlYXRpbmcgbGlzdCBpdGVtIGZyb20gdGV4dFwiLFxuICAgICAgY2FsbGJhY2s6ICgpID0+IHRoaXMuc2hvcnRjdXRMaXN0SXRlbXMoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQWx0XCJdLFxuICAgICAgICAgIGtleTogXCItXCIsXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICBpZDogXCJoZWFkaW5nLTFcIixcbiAgICAgIG5hbWU6IFwibGluZSBpbnRvIGxldmVsIDEgaGVhZGluZ1wiLFxuICAgICAgY2FsbGJhY2s6ICgpID0+IHRoaXMuc2hvcnRjdXRIZWFkZXIxKCksXG4gICAgICBob3RrZXlzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBtb2RpZmllcnM6IFtcIkN0cmxcIl0sXG4gICAgICAgICAga2V5OiBcIjFcIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgIGlkOiBcImhlYWRpbmctMlwiLFxuICAgICAgbmFtZTogXCJsaW5lIGludG8gbGV2ZWwgMiBoZWFkaW5nXCIsXG4gICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5zaG9ydGN1dEhlYWRlcjIoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQ3RybFwiXSxcbiAgICAgICAgICBrZXk6IFwiMlwiLFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwiaGVhZGluZy0zXCIsXG4gICAgICBuYW1lOiBcImxpbmUgaW50byBsZXZlbCAzIGhlYWRpbmdcIixcbiAgICAgIGNhbGxiYWNrOiAoKSA9PiB0aGlzLnNob3J0Y3V0SGVhZGVyMygpLFxuICAgICAgaG90a2V5czogW1xuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJDdHJsXCJdLFxuICAgICAgICAgIGtleTogXCIzXCIsXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICBpZDogXCJoZWFkaW5nLTRcIixcbiAgICAgIG5hbWU6IFwibGluZSBpbnRvIGxldmVsIDQgaGVhZGluZ1wiLFxuICAgICAgY2FsbGJhY2s6ICgpID0+IHRoaXMuc2hvcnRjdXRIZWFkZXI0KCksXG4gICAgICBob3RrZXlzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBtb2RpZmllcnM6IFtcIkN0cmxcIl0sXG4gICAgICAgICAga2V5OiBcIjRcIixcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgdGhpcy5hZGRDb21tYW5kKHtcbiAgICAgIGlkOiBcImhlYWRpbmctNVwiLFxuICAgICAgbmFtZTogXCJsaW5lIGludG8gbGV2ZWwgNSBoZWFkaW5nXCIsXG4gICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5zaG9ydGN1dEhlYWRlcjUoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQ3RybFwiXSxcbiAgICAgICAgICBrZXk6IFwiNVwiLFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICB0aGlzLmFkZENvbW1hbmQoe1xuICAgICAgaWQ6IFwiaGVhZGluZy02XCIsXG4gICAgICBuYW1lOiBcImxpbmUgaW50byBsZXZlbCA2IGhlYWRpbmdcIixcbiAgICAgIGNhbGxiYWNrOiAoKSA9PiB0aGlzLnNob3J0Y3V0SGVhZGVyNigpLFxuICAgICAgaG90a2V5czogW1xuICAgICAgICB7XG4gICAgICAgICAgbW9kaWZpZXJzOiBbXCJDdHJsXCJdLFxuICAgICAgICAgIGtleTogXCI2XCIsXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gIHRoaXMuYWRkQ29tbWFuZCh7XG4gICAgICBpZDogXCJoZWFkaW5nLTBcIixcbiAgICAgIG5hbWU6IFwiY2xlYXJpbmcgb2YgdGV4dCBmb3JtYXR0aW5nXCIsXG4gICAgICBjYWxsYmFjazogKCkgPT4gdGhpcy5zaG9ydGN1dEhlYWRlcjAoKSxcbiAgICAgIGhvdGtleXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIG1vZGlmaWVyczogW1wiQ3RybFwiXSxcbiAgICAgICAgICBrZXk6IFwiMFwiLFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcbiAgfVxuXG4gIGdldFNlbGVjdGVkVGV4dChlZGl0b3I6IGFueSkge1xuICAvL3RoYW5rcyB0byB1c2VyIFwiQXJnZW50aW5hIE9ydGVnYSBTw6FpbnpcIiBmcm9tIHRoZSBPYnNpZGlhbiBjb21tdW5pdHkgZm9yIHRoaXMgZmVhdHVyZVxuICAvL0ZvciBhIGxvbmcgdGltZSBJIHRyaWVkIHRvIGRvIHdpdGhvdXQgc3VjaCBhbiBhcHByb2FjaCwgd2hpY2ggcmVzdWx0ZWQgaW4gc2V2ZXJhbCBidWdzIGFuZCB0aGUgaW1wb3NzaWJpbGl0eSBvZiBmaXhpbmcgdGhlbSB3aXRoIG5vbi1jcnV0Y2hlc1xuICAgIGlmIChlZGl0b3Iuc29tZXRoaW5nU2VsZWN0ZWQoKSkge1xuICAgICAgbGV0IGN1cnNvclN0YXJ0ID0gZWRpdG9yLmdldEN1cnNvcih0cnVlKTtcbiAgICAgIGxldCBjdXJzb3JFbmQgPSBlZGl0b3IuZ2V0Q3Vyc29yKGZhbHNlKTtcbiAgICAgIGxldCBjb250ZW50ID0gZWRpdG9yLmdldFJhbmdlKFxuICAgICAgICB7IGxpbmU6IGN1cnNvclN0YXJ0LmxpbmUsIGNoOiAwIH0sXG4gICAgICAgIHsgbGluZTogY3Vyc29yRW5kLmxpbmUsIGNoOiBlZGl0b3IuZ2V0TGluZShjdXJzb3JFbmQubGluZSkubGVuZ3RoIH1cbiAgICAgICk7XG4gICAgICByZXR1cm4ge1xuICAgICAgICBzdGFydDogeyBsaW5lOiBjdXJzb3JTdGFydC5saW5lLCBjaDogMCB9LFxuICAgICAgICBlbmQ6IHtcbiAgICAgICAgICBsaW5lOiBjdXJzb3JFbmQubGluZSxcbiAgICAgICAgICBjaDogZWRpdG9yLmdldExpbmUoY3Vyc29yRW5kLmxpbmUpLmxlbmd0aCxcbiAgICAgICAgfSxcbiAgICAgICAgY29udGVudDogY29udGVudCxcbiAgICAgIH07XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIFRvZ2dsZSB0aGUgdG9kbyBpbiB0aGUgbGluZVxuICAgICAgdmFyIGxpbmVOciA9IGVkaXRvci5nZXRDdXJzb3IoKS5saW5lO1xuICAgICAgdmFyIGNvbnRlbnRzID0gZWRpdG9yLmdldERvYygpLmdldExpbmUobGluZU5yKTtcbiAgICAgIGxldCBjdXJzb3JTdGFydCA9IHtcbiAgICAgICAgbGluZTogbGluZU5yLFxuICAgICAgICBjaDogMCxcbiAgICAgIH07XG4gICAgICBsZXQgY3Vyc29yRW5kID0ge1xuICAgICAgICBsaW5lOiBsaW5lTnIsXG4gICAgICAgIGNoOiBjb250ZW50cy5sZW5ndGgsXG4gICAgICB9O1xuICAgICAgbGV0IGNvbnRlbnQgPSBlZGl0b3IuZ2V0UmFuZ2UoY3Vyc29yU3RhcnQsIGN1cnNvckVuZCk7XG4gICAgICByZXR1cm4geyBzdGFydDogY3Vyc29yU3RhcnQsIGVuZDogY3Vyc29yRW5kLCBjb250ZW50OiBjb250ZW50IH07XG4gICAgfVxuICB9XG5cbiAgc2hvcnRjdXRFeGNsYW1hdGlvbk1hcmsoKTogdm9pZCB7XG4gICAgbGV0IGFjdGl2ZUxlYWY6IGFueSA9IHRoaXMuYXBwLndvcmtzcGFjZS5hY3RpdmVMZWFmO1xuICAgIGxldCBlZGl0b3IgPSBhY3RpdmVMZWFmLnZpZXcuc291cmNlTW9kZS5jbUVkaXRvcjtcbiAgICBsZXQgc2VsZWN0ZWRUZXh0ID0gZWRpdG9yLnNvbWV0aGluZ1NlbGVjdGVkKClcbiAgICAgID8gZWRpdG9yLmdldFNlbGVjdGlvbigpXG4gICAgICA6IGZhbHNlO1xuICAgIGVkaXRvci5yZXBsYWNlU2VsZWN0aW9uKGBcXCFgKTtcbiAgfVxuXG4gIHNob3J0Y3V0QXQoKTogdm9pZCB7XG4gICAgbGV0IGFjdGl2ZUxlYWY6IGFueSA9IHRoaXMuYXBwLndvcmtzcGFjZS5hY3RpdmVMZWFmO1xuICAgIGxldCBlZGl0b3IgPSBhY3RpdmVMZWFmLnZpZXcuc291cmNlTW9kZS5jbUVkaXRvcjtcbiAgICBsZXQgc2VsZWN0ZWRUZXh0ID0gZWRpdG9yLnNvbWV0aGluZ1NlbGVjdGVkKClcbiAgICAgID8gZWRpdG9yLmdldFNlbGVjdGlvbigpXG4gICAgICA6IGZhbHNlO1xuICAgIGVkaXRvci5yZXBsYWNlU2VsZWN0aW9uKGBAYCk7XG4gIH1cblxuICBzaG9ydGN1dEhhc2goKTogdm9pZCB7XG4gICAgbGV0IGFjdGl2ZUxlYWY6IGFueSA9IHRoaXMuYXBwLndvcmtzcGFjZS5hY3RpdmVMZWFmO1xuICAgIGxldCBlZGl0b3IgPSBhY3RpdmVMZWFmLnZpZXcuc291cmNlTW9kZS5jbUVkaXRvcjtcbiAgICBsZXQgc2VsZWN0ZWRUZXh0ID0gZWRpdG9yLnNvbWV0aGluZ1NlbGVjdGVkKClcbiAgICAgID8gZWRpdG9yLmdldFNlbGVjdGlvbigpXG4gICAgICA6IGZhbHNlO1xuICAgIGVkaXRvci5yZXBsYWNlU2VsZWN0aW9uKGAjYCk7XG4gIH1cblxuICBzaG9ydGN1dERvbGxhcigpOiB2b2lkIHtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSBlZGl0b3Iuc29tZXRoaW5nU2VsZWN0ZWQoKVxuICAgICAgPyBlZGl0b3IuZ2V0U2VsZWN0aW9uKClcbiAgICAgIDogZmFsc2U7XG4gICAgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYCRgKTtcbiAgfVxuXG4gIHNob3J0Y3V0UGVyY2VudCgpOiB2b2lkIHtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSBlZGl0b3Iuc29tZXRoaW5nU2VsZWN0ZWQoKVxuICAgICAgPyBlZGl0b3IuZ2V0U2VsZWN0aW9uKClcbiAgICAgIDogZmFsc2U7XG4gICAgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYFxcJWApO1xuICB9XG5cbiAgc2hvcnRjdXRDaXJjdW1mbGV4KCk6IHZvaWQge1xuICAgIGxldCBhY3RpdmVMZWFmOiBhbnkgPSB0aGlzLmFwcC53b3Jrc3BhY2UuYWN0aXZlTGVhZjtcbiAgICBsZXQgZWRpdG9yID0gYWN0aXZlTGVhZi52aWV3LnNvdXJjZU1vZGUuY21FZGl0b3I7XG4gICAgbGV0IHNlbGVjdGVkVGV4dCA9IGVkaXRvci5zb21ldGhpbmdTZWxlY3RlZCgpXG4gICAgICA/IGVkaXRvci5nZXRTZWxlY3Rpb24oKVxuICAgICAgOiBmYWxzZTtcbiAgICBlZGl0b3IucmVwbGFjZVNlbGVjdGlvbihgXFxeYCk7XG4gIH1cblxuICBzaG9ydGN1dEFtcGVyc2FuZCgpOiB2b2lkIHtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSBlZGl0b3Iuc29tZXRoaW5nU2VsZWN0ZWQoKVxuICAgICAgPyBlZGl0b3IuZ2V0U2VsZWN0aW9uKClcbiAgICAgIDogZmFsc2U7XG4gICAgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYFxcJmApO1xuICB9XG5cbiAgc2hvcnRjdXRMZXNzVGhhbigpOiB2b2lkIHtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSBlZGl0b3Iuc29tZXRoaW5nU2VsZWN0ZWQoKVxuICAgICAgPyBlZGl0b3IuZ2V0U2VsZWN0aW9uKClcbiAgICAgIDogZmFsc2U7XG4gICAgaWYgKHNlbGVjdGVkVGV4dCkge1xuICAgICAgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYDwke3NlbGVjdGVkVGV4dH0+YCk7XG4gICAgfSBlbHNlIGVkaXRvci5yZXBsYWNlU2VsZWN0aW9uKGA8YCk7XG4gIH1cblxuICBzaG9ydGN1dEdyZWF0ZXJUaGFuKCk6IHZvaWQge1xuXHRcdGxldCBhY3RpdmVMZWFmOiBhbnkgPSB0aGlzLmFwcC53b3Jrc3BhY2UuYWN0aXZlTGVhZjtcbiAgICBsZXQgZWRpdG9yID0gYWN0aXZlTGVhZi52aWV3LnNvdXJjZU1vZGUuY21FZGl0b3I7XG4gICAgbGV0IHNlbGVjdGVkVGV4dCA9IHRoaXMuZ2V0U2VsZWN0ZWRUZXh0KGVkaXRvcik7XG5cbiAgICBsZXQgcmVzdWx0VGV4dCA9IFwiPiBcIiArIHNlbGVjdGVkVGV4dC5jb250ZW50LnNwbGl0KCdcXG4nKS5qb2luKFwiXFxuPiBcIik7XG5cdFx0ZWRpdG9yLnJlcGxhY2VSYW5nZShyZXN1bHRUZXh0LCBzZWxlY3RlZFRleHQuc3RhcnQsIHNlbGVjdGVkVGV4dC5lbmQpO1xuICB9XG5cbiAgc2hvcnRjdXRMZWZ0U3F1YXJlQnJhY2tldCgpOiB2b2lkIHtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSBlZGl0b3Iuc29tZXRoaW5nU2VsZWN0ZWQoKVxuICAgICAgPyBlZGl0b3IuZ2V0U2VsZWN0aW9uKClcbiAgICAgIDogZmFsc2U7XG4gICAgaWYgKHNlbGVjdGVkVGV4dCkge1xuICAgICAgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYFtbJHtzZWxlY3RlZFRleHR9XV1gKTtcbiAgICB9IGVsc2UgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYFtgKTtcbiAgfVxuXG4gIHNob3J0Y3V0UmlnaHRTcXVhcmVCcmFja2V0KCk6IHZvaWQge1xuICAgIGxldCBhY3RpdmVMZWFmOiBhbnkgPSB0aGlzLmFwcC53b3Jrc3BhY2UuYWN0aXZlTGVhZjtcbiAgICBsZXQgZWRpdG9yID0gYWN0aXZlTGVhZi52aWV3LnNvdXJjZU1vZGUuY21FZGl0b3I7XG4gICAgbGV0IHNlbGVjdGVkVGV4dCA9IGVkaXRvci5zb21ldGhpbmdTZWxlY3RlZCgpXG4gICAgICA/IGVkaXRvci5nZXRTZWxlY3Rpb24oKVxuICAgICAgOiBmYWxzZTtcbiAgICBlZGl0b3IucmVwbGFjZVNlbGVjdGlvbihgXWApO1xuICB9XG5cbiAgc2hvcnRjdXRQeUNvZGUoKTogdm9pZCB7XG4gICAgbGV0IGFjdGl2ZUxlYWY6IGFueSA9IHRoaXMuYXBwLndvcmtzcGFjZS5hY3RpdmVMZWFmO1xuICAgIGxldCBlZGl0b3IgPSBhY3RpdmVMZWFmLnZpZXcuc291cmNlTW9kZS5jbUVkaXRvcjtcbiAgICBsZXQgc2VsZWN0ZWRUZXh0ID0gZWRpdG9yLnNvbWV0aGluZ1NlbGVjdGVkKClcbiAgICAgID8gZWRpdG9yLmdldFNlbGVjdGlvbigpXG4gICAgICA6IGZhbHNlO1xuICAgIGlmIChzZWxlY3RlZFRleHQpIHtcbiAgICAgIGVkaXRvci5yZXBsYWNlU2VsZWN0aW9uKGBcXGBcXGBcXGBweVxcbiR7c2VsZWN0ZWRUZXh0fVxcblxcYFxcYFxcYGApO1xuICAgIH0gZWxzZSBlZGl0b3IucmVwbGFjZVNlbGVjdGlvbihgXFxgYCk7XG4gIH1cblxuICBzaG9ydGN1dENvZGVGZW5jZXMoKTogdm9pZCB7XG4gICAgbGV0IGFjdGl2ZUxlYWY6IGFueSA9IHRoaXMuYXBwLndvcmtzcGFjZS5hY3RpdmVMZWFmO1xuICAgIGxldCBlZGl0b3IgPSBhY3RpdmVMZWFmLnZpZXcuc291cmNlTW9kZS5jbUVkaXRvcjtcbiAgICBsZXQgc2VsZWN0ZWRUZXh0ID0gZWRpdG9yLnNvbWV0aGluZ1NlbGVjdGVkKClcbiAgICAgID8gZWRpdG9yLmdldFNlbGVjdGlvbigpXG4gICAgICA6IGZhbHNlO1xuICAgIGlmIChzZWxlY3RlZFRleHQpIHtcbiAgICAgIGVkaXRvci5yZXBsYWNlU2VsZWN0aW9uKGBcXGAke3NlbGVjdGVkVGV4dH1cXGBgKTtcbiAgICB9IGVsc2UgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYFxcYGApO1xuICB9XG5cbiAgc2hvcnRjdXRSaWdodEN1cmx5QnJhY2tldCgpOiB2b2lkIHtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSBlZGl0b3Iuc29tZXRoaW5nU2VsZWN0ZWQoKVxuICAgICAgPyBlZGl0b3IuZ2V0U2VsZWN0aW9uKClcbiAgICAgIDogZmFsc2U7XG4gICAgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYH1gKTtcbiAgfVxuXG4gIHNob3J0Y3V0TGVmdEN1cmx5QnJhY2tldCgpOiB2b2lkIHtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSBlZGl0b3Iuc29tZXRoaW5nU2VsZWN0ZWQoKVxuICAgICAgPyBlZGl0b3IuZ2V0U2VsZWN0aW9uKClcbiAgICAgIDogZmFsc2U7XG4gICAgaWYgKHNlbGVjdGVkVGV4dCkge1xuICAgICAgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYFxceyR7c2VsZWN0ZWRUZXh0fVxcfWApO1xuICAgIH0gZWxzZSBlZGl0b3IucmVwbGFjZVNlbGVjdGlvbihgXFx7YCk7XG4gIH1cblxuICBzaG9ydGN1dFZlcnRpY2FsTGluZSgpOiB2b2lkIHtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSBlZGl0b3Iuc29tZXRoaW5nU2VsZWN0ZWQoKVxuICAgICAgPyBlZGl0b3IuZ2V0U2VsZWN0aW9uKClcbiAgICAgIDogZmFsc2U7XG4gICAgZWRpdG9yLnJlcGxhY2VTZWxlY3Rpb24oYHxgKTtcbiAgfVxuXG4gIHNob3J0Y3V0TGlzdEl0ZW1zKCk6IHZvaWQge1xuICAgIGxldCBhY3RpdmVMZWFmOiBhbnkgPSB0aGlzLmFwcC53b3Jrc3BhY2UuYWN0aXZlTGVhZjtcbiAgICBsZXQgZWRpdG9yID0gYWN0aXZlTGVhZi52aWV3LnNvdXJjZU1vZGUuY21FZGl0b3I7XG4gICAgbGV0IHNlbGVjdGVkVGV4dCA9IHRoaXMuZ2V0U2VsZWN0ZWRUZXh0KGVkaXRvcik7XG5cbiAgICBsZXQgcmVzdWx0VGV4dCA9IFwiLSBcIiArIHNlbGVjdGVkVGV4dC5jb250ZW50LnNwbGl0KCdcXG4nKS5qb2luKFwiXFxuLSBcIik7XG5cbiAgICBsZXQgcmVzdWx0VGV4dENoZWNrID0gcmVzdWx0VGV4dDtcbiAgICB3aGlsZSAocmVzdWx0VGV4dENoZWNrLnNlYXJjaChcIi0gLSBcIikgPj0gMCkge1xuICAgICAgcmVzdWx0VGV4dENoZWNrID0gcmVzdWx0VGV4dENoZWNrLnJlcGxhY2UoXCItIC0gXCIsIFwiXHQtIFwiKTtcbiAgICB9XG4gICAgd2hpbGUgKHJlc3VsdFRleHRDaGVjay5zZWFyY2goXCItIFx0XHRcIikgPj0gMCkge1xuICAgICAgcmVzdWx0VGV4dENoZWNrID0gcmVzdWx0VGV4dENoZWNrLnJlcGxhY2UoXCItIFx0XHRcIiwgXCJcdC0gXHRcIik7XG4gICAgfVxuICAgIHdoaWxlIChyZXN1bHRUZXh0Q2hlY2suc2VhcmNoKFwiLSBcdC0gXCIpID49IDApIHtcbiAgICAgIHJlc3VsdFRleHRDaGVjayA9IHJlc3VsdFRleHRDaGVjay5yZXBsYWNlKFwiLSBcdC0gXCIsIFwiXHRcdC0gXCIpO1xuICAgIH1cbiAgICBlZGl0b3IucmVwbGFjZVJhbmdlKHJlc3VsdFRleHRDaGVjaywgc2VsZWN0ZWRUZXh0LnN0YXJ0LCBzZWxlY3RlZFRleHQuZW5kKTtcbiAgfVxuXG4gIHNob3J0Y3V0SGVhZGVyMCgpOiB2b2lkIHtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSB0aGlzLmdldFNlbGVjdGVkVGV4dChlZGl0b3IpO1xuXG4gICAgdmFyIGo7XG4gICAgbGV0IHNwbGl0dGVkU2VsZWN0ZWRUZXh0ID0gc2VsZWN0ZWRUZXh0LmNvbnRlbnQuc3BsaXQoYFxcbmApO1xuICAgIGZvciAoaiBpbiBzcGxpdHRlZFNlbGVjdGVkVGV4dCkge1xuICAgICAgd2hpbGUgKHNwbGl0dGVkU2VsZWN0ZWRUZXh0W2pdLnN1YnN0cmluZygwLDIpID09IFwiIyNcIikge1xuICAgICAgICBzcGxpdHRlZFNlbGVjdGVkVGV4dFtqXSA9IHNwbGl0dGVkU2VsZWN0ZWRUZXh0W2pdLnJlcGxhY2UoXCIjI1wiLCBcIiNcIik7XG4gICAgICB9XG4gICAgICB3aGlsZSAoc3BsaXR0ZWRTZWxlY3RlZFRleHRbal0uc3Vic3RyaW5nKDAsMikgPT0gXCIjIFwiKSB7XG4gICAgICAgIHNwbGl0dGVkU2VsZWN0ZWRUZXh0W2pdID0gc3BsaXR0ZWRTZWxlY3RlZFRleHRbal0uc3Vic3RyKDIpO1xuICAgICAgfVxuICAgICAgd2hpbGUgKHNwbGl0dGVkU2VsZWN0ZWRUZXh0W2pdLnN1YnN0cmluZygwLDIpID09IFwiPiBcIikge1xuICAgICAgICBzcGxpdHRlZFNlbGVjdGVkVGV4dFtqXSA9IHNwbGl0dGVkU2VsZWN0ZWRUZXh0W2pdLnN1YnN0cigyKTtcbiAgICAgIH1cbiAgICAgIHdoaWxlIChzcGxpdHRlZFNlbGVjdGVkVGV4dFtqXS5zdWJzdHJpbmcoMCwyKSA9PSBcIlx0XHRcIikge1xuICAgICAgICBzcGxpdHRlZFNlbGVjdGVkVGV4dFtqXSA9IHNwbGl0dGVkU2VsZWN0ZWRUZXh0W2pdLnJlcGxhY2UoXCJcdFx0XCIsIFwiXHRcIik7XG4gICAgICB9XG4gICAgICB3aGlsZSAoc3BsaXR0ZWRTZWxlY3RlZFRleHRbal0uc3Vic3RyaW5nKDAsMykgPT0gXCJcdC0gXCIpIHtcbiAgICAgICAgc3BsaXR0ZWRTZWxlY3RlZFRleHRbal0gPSBzcGxpdHRlZFNlbGVjdGVkVGV4dFtqXS5zdWJzdHIoMyk7XG4gICAgICB9XG4gICAgICB3aGlsZSAoc3BsaXR0ZWRTZWxlY3RlZFRleHRbal0uc3Vic3RyaW5nKDAsMikgPT0gXCItIFwiKSB7XG4gICAgICAgIHNwbGl0dGVkU2VsZWN0ZWRUZXh0W2pdID0gc3BsaXR0ZWRTZWxlY3RlZFRleHRbal0uc3Vic3RyKDIpO1xuICAgICAgfVxuICAgICAgdmFyIHJlX2RpZ2l0cyA9IC9eXFxkXFwuXFxzLztcbiAgICAgIHNwbGl0dGVkU2VsZWN0ZWRUZXh0W2pdID0gc3BsaXR0ZWRTZWxlY3RlZFRleHRbal0ucmVwbGFjZShyZV9kaWdpdHMsIFwiXCIpO1xuICAgIH1cblxuICAgIGVkaXRvci5yZXBsYWNlUmFuZ2Uoc3BsaXR0ZWRTZWxlY3RlZFRleHQsIHNlbGVjdGVkVGV4dC5zdGFydCwgc2VsZWN0ZWRUZXh0LmVuZCk7XG4gIH1cblxuICBzaG9ydGN1dEhlYWRlcjEoKTogdm9pZCB7XG4gICAgdGhpcy5zaG9ydGN1dEhlYWRlcjAoKTtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSB0aGlzLmdldFNlbGVjdGVkVGV4dChlZGl0b3IpO1xuXG4gICAgLy92YXIgY2xlYXJUZXh0ID0gc2VsZWN0ZWRUZXh0LmNvbnRlbnQuc2hvcnRjdXRIZWFkZXIwKClcbiAgICB2YXIgcmVzdWx0VGV4dCA9IFwiIyBcIiArIHNlbGVjdGVkVGV4dC5jb250ZW50O1xuICAgIGVkaXRvci5yZXBsYWNlUmFuZ2UocmVzdWx0VGV4dCwgc2VsZWN0ZWRUZXh0LnN0YXJ0LCBzZWxlY3RlZFRleHQuZW5kKTtcbiAgfVxuXG4gIHNob3J0Y3V0SGVhZGVyMigpOiB2b2lkIHtcbiAgICB0aGlzLnNob3J0Y3V0SGVhZGVyMCgpO1xuICAgIGxldCBhY3RpdmVMZWFmOiBhbnkgPSB0aGlzLmFwcC53b3Jrc3BhY2UuYWN0aXZlTGVhZjtcbiAgICBsZXQgZWRpdG9yID0gYWN0aXZlTGVhZi52aWV3LnNvdXJjZU1vZGUuY21FZGl0b3I7XG4gICAgbGV0IHNlbGVjdGVkVGV4dCA9IHRoaXMuZ2V0U2VsZWN0ZWRUZXh0KGVkaXRvcik7XG5cbiAgICB2YXIgcmVzdWx0VGV4dCA9IFwiIyMgXCIgKyBzZWxlY3RlZFRleHQuY29udGVudDtcbiAgICBlZGl0b3IucmVwbGFjZVJhbmdlKHJlc3VsdFRleHQsIHNlbGVjdGVkVGV4dC5zdGFydCwgc2VsZWN0ZWRUZXh0LmVuZCk7XG4gIH1cblxuICBzaG9ydGN1dEhlYWRlcjMoKTogdm9pZCB7XG4gICAgdGhpcy5zaG9ydGN1dEhlYWRlcjAoKTtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSB0aGlzLmdldFNlbGVjdGVkVGV4dChlZGl0b3IpO1xuXG4gICAgdmFyIHJlc3VsdFRleHQgPSBcIiMjIyBcIiArIHNlbGVjdGVkVGV4dC5jb250ZW50O1xuICAgIGVkaXRvci5yZXBsYWNlUmFuZ2UocmVzdWx0VGV4dCwgc2VsZWN0ZWRUZXh0LnN0YXJ0LCBzZWxlY3RlZFRleHQuZW5kKTtcbiAgfVxuXG4gIHNob3J0Y3V0SGVhZGVyNCgpOiB2b2lkIHtcbiAgICB0aGlzLnNob3J0Y3V0SGVhZGVyMCgpO1xuICAgIGxldCBhY3RpdmVMZWFmOiBhbnkgPSB0aGlzLmFwcC53b3Jrc3BhY2UuYWN0aXZlTGVhZjtcbiAgICBsZXQgZWRpdG9yID0gYWN0aXZlTGVhZi52aWV3LnNvdXJjZU1vZGUuY21FZGl0b3I7XG4gICAgbGV0IHNlbGVjdGVkVGV4dCA9IHRoaXMuZ2V0U2VsZWN0ZWRUZXh0KGVkaXRvcik7XG5cbiAgICB2YXIgcmVzdWx0VGV4dCA9IFwiIyMjIyBcIiArIHNlbGVjdGVkVGV4dC5jb250ZW50O1xuICAgIGVkaXRvci5yZXBsYWNlUmFuZ2UocmVzdWx0VGV4dCwgc2VsZWN0ZWRUZXh0LnN0YXJ0LCBzZWxlY3RlZFRleHQuZW5kKTtcbiAgfVxuXG4gIHNob3J0Y3V0SGVhZGVyNSgpOiB2b2lkIHtcbiAgICB0aGlzLnNob3J0Y3V0SGVhZGVyMCgpO1xuICAgIGxldCBhY3RpdmVMZWFmOiBhbnkgPSB0aGlzLmFwcC53b3Jrc3BhY2UuYWN0aXZlTGVhZjtcbiAgICBsZXQgZWRpdG9yID0gYWN0aXZlTGVhZi52aWV3LnNvdXJjZU1vZGUuY21FZGl0b3I7XG4gICAgbGV0IHNlbGVjdGVkVGV4dCA9IHRoaXMuZ2V0U2VsZWN0ZWRUZXh0KGVkaXRvcik7XG5cbiAgICB2YXIgcmVzdWx0VGV4dCA9IFwiIyMjIyMgXCIgKyBzZWxlY3RlZFRleHQuY29udGVudDtcbiAgICBlZGl0b3IucmVwbGFjZVJhbmdlKHJlc3VsdFRleHQsIHNlbGVjdGVkVGV4dC5zdGFydCwgc2VsZWN0ZWRUZXh0LmVuZCk7XG4gIH1cblxuICBzaG9ydGN1dEhlYWRlcjYoKTogdm9pZCB7XG4gICAgdGhpcy5zaG9ydGN1dEhlYWRlcjAoKTtcbiAgICBsZXQgYWN0aXZlTGVhZjogYW55ID0gdGhpcy5hcHAud29ya3NwYWNlLmFjdGl2ZUxlYWY7XG4gICAgbGV0IGVkaXRvciA9IGFjdGl2ZUxlYWYudmlldy5zb3VyY2VNb2RlLmNtRWRpdG9yO1xuICAgIGxldCBzZWxlY3RlZFRleHQgPSB0aGlzLmdldFNlbGVjdGVkVGV4dChlZGl0b3IpO1xuXG4gICAgdmFyIHJlc3VsdFRleHQgPSBcIiMjIyMjIyBcIiArIHNlbGVjdGVkVGV4dC5jb250ZW50O1xuICAgIGVkaXRvci5yZXBsYWNlUmFuZ2UocmVzdWx0VGV4dCwgc2VsZWN0ZWRUZXh0LnN0YXJ0LCBzZWxlY3RlZFRleHQuZW5kKTtcbiAgfVxuXG5cblxufVxuIl0sIm5hbWVzIjpbIlBsdWdpbiJdLCJtYXBwaW5ncyI6Ijs7OztBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsSUFBSSxhQUFhLEdBQUcsU0FBUyxDQUFDLEVBQUUsQ0FBQyxFQUFFO0FBQ25DLElBQUksYUFBYSxHQUFHLE1BQU0sQ0FBQyxjQUFjO0FBQ3pDLFNBQVMsRUFBRSxTQUFTLEVBQUUsRUFBRSxFQUFFLFlBQVksS0FBSyxJQUFJLFVBQVUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxTQUFTLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNwRixRQUFRLFVBQVUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEtBQUssSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLElBQUksTUFBTSxDQUFDLFNBQVMsQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUMxRyxJQUFJLE9BQU8sYUFBYSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztBQUMvQixDQUFDLENBQUM7QUFDRjtBQUNPLFNBQVMsU0FBUyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUU7QUFDaEMsSUFBSSxhQUFhLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0FBQ3hCLElBQUksU0FBUyxFQUFFLEdBQUcsRUFBRSxJQUFJLENBQUMsV0FBVyxHQUFHLENBQUMsQ0FBQyxFQUFFO0FBQzNDLElBQUksQ0FBQyxDQUFDLFNBQVMsR0FBRyxDQUFDLEtBQUssSUFBSSxHQUFHLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDLFNBQVMsR0FBRyxDQUFDLENBQUMsU0FBUyxFQUFFLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQztBQUN6RixDQUFDO0FBdUNEO0FBQ08sU0FBUyxTQUFTLENBQUMsT0FBTyxFQUFFLFVBQVUsRUFBRSxDQUFDLEVBQUUsU0FBUyxFQUFFO0FBQzdELElBQUksU0FBUyxLQUFLLENBQUMsS0FBSyxFQUFFLEVBQUUsT0FBTyxLQUFLLFlBQVksQ0FBQyxHQUFHLEtBQUssR0FBRyxJQUFJLENBQUMsQ0FBQyxVQUFVLE9BQU8sRUFBRSxFQUFFLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFO0FBQ2hILElBQUksT0FBTyxLQUFLLENBQUMsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDLEVBQUUsVUFBVSxPQUFPLEVBQUUsTUFBTSxFQUFFO0FBQy9ELFFBQVEsU0FBUyxTQUFTLENBQUMsS0FBSyxFQUFFLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRTtBQUNuRyxRQUFRLFNBQVMsUUFBUSxDQUFDLEtBQUssRUFBRSxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsU0FBUyxDQUFDLE9BQU8sQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRTtBQUN0RyxRQUFRLFNBQVMsSUFBSSxDQUFDLE1BQU0sRUFBRSxFQUFFLE1BQU0sQ0FBQyxJQUFJLEdBQUcsT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsR0FBRyxLQUFLLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLEVBQUUsUUFBUSxDQUFDLENBQUMsRUFBRTtBQUN0SCxRQUFRLElBQUksQ0FBQyxDQUFDLFNBQVMsR0FBRyxTQUFTLENBQUMsS0FBSyxDQUFDLE9BQU8sRUFBRSxVQUFVLElBQUksRUFBRSxDQUFDLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQztBQUM5RSxLQUFLLENBQUMsQ0FBQztBQUNQLENBQUM7QUFDRDtBQUNPLFNBQVMsV0FBVyxDQUFDLE9BQU8sRUFBRSxJQUFJLEVBQUU7QUFDM0MsSUFBSSxJQUFJLENBQUMsR0FBRyxFQUFFLEtBQUssRUFBRSxDQUFDLEVBQUUsSUFBSSxFQUFFLFdBQVcsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsSUFBSSxFQUFFLEVBQUUsRUFBRSxHQUFHLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDO0FBQ3JILElBQUksT0FBTyxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLE9BQU8sRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsUUFBUSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLE9BQU8sTUFBTSxLQUFLLFVBQVUsS0FBSyxDQUFDLENBQUMsTUFBTSxDQUFDLFFBQVEsQ0FBQyxHQUFHLFdBQVcsRUFBRSxPQUFPLElBQUksQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUM7QUFDN0osSUFBSSxTQUFTLElBQUksQ0FBQyxDQUFDLEVBQUUsRUFBRSxPQUFPLFVBQVUsQ0FBQyxFQUFFLEVBQUUsT0FBTyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRTtBQUN0RSxJQUFJLFNBQVMsSUFBSSxDQUFDLEVBQUUsRUFBRTtBQUN0QixRQUFRLElBQUksQ0FBQyxFQUFFLE1BQU0sSUFBSSxTQUFTLENBQUMsaUNBQWlDLENBQUMsQ0FBQztBQUN0RSxRQUFRLE9BQU8sQ0FBQyxFQUFFLElBQUk7QUFDdEIsWUFBWSxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxRQUFRLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLElBQUksRUFBRSxPQUFPLENBQUMsQ0FBQztBQUN6SyxZQUFZLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7QUFDcEQsWUFBWSxRQUFRLEVBQUUsQ0FBQyxDQUFDLENBQUM7QUFDekIsZ0JBQWdCLEtBQUssQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLE1BQU07QUFDOUMsZ0JBQWdCLEtBQUssQ0FBQyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLElBQUksRUFBRSxLQUFLLEVBQUUsQ0FBQztBQUN4RSxnQkFBZ0IsS0FBSyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsU0FBUztBQUNqRSxnQkFBZ0IsS0FBSyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsU0FBUztBQUNqRSxnQkFBZ0I7QUFDaEIsb0JBQW9CLElBQUksRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLE1BQU0sR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxTQUFTLEVBQUU7QUFDaEksb0JBQW9CLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLEtBQUssR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLEVBQUU7QUFDMUcsb0JBQW9CLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLE1BQU0sRUFBRTtBQUN6RixvQkFBb0IsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFD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
