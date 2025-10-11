'use client';

import { useState } from 'react';
import { Copy, Check } from 'lucide-react';

export function Terminal_Prompt() {
  const [copied, setCopied] = useState(false);
  
  const terminalSteps = [
    { line: 'You are a helpful assistant. For each user query, generate a set of five responses. Each response should be approximately 200 words.', showPrompt: true },
    { line: 'Return the responses each within a separate <response> tag.', showPrompt: false },
    { line: 'Each <response> tag include a <text> and a numeric <probability>.', showPrompt: false },
    { line: 'Please sample at random from the full distribution.', showPrompt: false },
    { line: '<user_query>Write a short story about a bear.</user_query>', showPrompt: true },
  ];

  const fullPrompt = terminalSteps.map(step => step.line).join('\n');

  const copyToClipboard = () => {
    navigator.clipboard.writeText(fullPrompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full rounded-lg shadow-lg overflow-hidden bg-gray-900 text-white font-mono text-sm relative min-h-[300px]">
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