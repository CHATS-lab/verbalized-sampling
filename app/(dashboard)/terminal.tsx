'use client';

import { useState, useEffect } from 'react';
import { Copy, Check, RotateCcw } from 'lucide-react';

export function Terminal() {
  const [terminalStep, setTerminalStep] = useState(0);
  const [copied, setCopied] = useState(false);
  const terminalSteps = [
    { line: 'Generate 10 responses to the user query. Each response should be approximately 200 words.', showPrompt: true },
    { line: '', showPrompt: false },
    { line: 'Return the responses in JSON format with the key: "responses" (list of dicts). Each dictionary must include:', showPrompt: false },
    { line: '- \'text\': the response string only (no explanation or extra text).', showPrompt: false },
    { line: '- \'probability\': the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution).', showPrompt: false },
    { line: '', showPrompt: false },
    { line: 'Randomly sample the responses from the full distribution. Return ONLY the JSON object, with no additional explanations or text.', showPrompt: false },
    { line: '', showPrompt: false },
    { line: '<user_query>Write a short story about a bear.</user_query>', showPrompt: true },
  ];

  const fullPrompt = terminalSteps.map(step => step.line).join('\n');

  useEffect(() => {
    if (terminalStep < terminalSteps.length) {
      const timer = setTimeout(() => {
        setTerminalStep(prev => prev + 1);
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [terminalStep, terminalSteps.length]);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(fullPrompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const restartAnimation = () => {
    setTerminalStep(0);
  };

  return (
    <div className="w-full rounded-lg shadow-lg overflow-hidden bg-gray-900 text-white font-mono text-sm relative min-h-[400px]">
      <div className="p-4">
        <div className="flex justify-between items-center mb-4">
          <div className="flex space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={restartAnimation}
              className="text-gray-400 hover:text-white transition-colors"
              aria-label="Restart animation"
            >
              <RotateCcw className="h-5 w-5" />
            </button>
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
        <div className="space-y-2">
          {terminalSteps.map((step, index) => (
            <div
              key={index}
              className={`${index > terminalStep ? 'opacity-0' : 'opacity-100'} transition-opacity duration-300`}
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
