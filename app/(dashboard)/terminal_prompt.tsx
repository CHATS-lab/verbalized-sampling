'use client';

import { useState } from 'react';
import { Copy, Check } from 'lucide-react';

export function Terminal_Prompt() {
  const [copied, setCopied] = useState(false);
  
  // const terminalSteps = [
  //   { line: 'Generate 10 responses to the user query, each within a separate <response> tag.', showPrompt: true },
  //   { line: 'Each <response> tag must include a <text> and a numeric <probability>.', showPrompt: false },
  //   { line: 'Randomly sample the responses from the full distribution.', showPrompt: false },
  //   { line: '<user_query>Write a 100-word story about a bear.</user_query>', showPrompt: true },
  // ];
  const terminalSteps = [
    { line: 'Generate 10 responses to the user query, each within a separate <response> tag.', showPrompt: true },
    { line: 'Each <response> tag must include a <text> and a numeric <probability>.', showPrompt: false },
    { line: 'Randomly sample the responses from the full distribution.', showPrompt: false },
    { line: '<user_query>Write a 100-word story about a bear.</user_query>', showPrompt: true },
  ];

  const fullPrompt = terminalSteps.map(step => step.line).join('\n');

  const copyToClipboard = () => {
    navigator.clipboard.writeText(fullPrompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full rounded-lg shadow-lg overflow-hidden bg-gray-900 text-white font-mono text-sm relative min-h-[200px]">
      <div className="p-4">
        <div className="flex justify-between items-center mb-4">
          <div className="flex space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={copyToClipboard}
              className="text-gray-400 hover:text-white transition-colors"
              aria-label="Copy to clipboard"
            >
              {copied ? (
                <Check className="h-5 w-5" />
              ) : (
                <Copy className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>
        <div className="font-mono text-sm">
          {terminalSteps.map((step, index) => (
            <div
              key={index}
              className="block mb-1"
            >
              {step.showPrompt && <span className="text-green-400">$ </span>}
              {step.line}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}