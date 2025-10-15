'use client';

import { useState, useEffect } from 'react';
import { Copy, Check } from 'lucide-react';

interface TerminalProps {
  height?: string;
}

export function Terminal({ height = "auto" }: TerminalProps) {
  const pipCommand = [
    'pip install verbalized-sampling'
  ];

  const pythonCode = [
    'verbalize run',
    '    --task joke \\',
    '    --prompt "Tell me a joke about coffee." \\',
    '    --model "gpt-4.1" \\',
    '    --methods "direct vs_standard" \\',
    '    --num-responses 10 \\',
    '    --metrics "diversity joke_quality"'
  ];

  const allSteps = [...pipCommand, ...pythonCode];
  const [terminalStep, setTerminalStep] = useState(allSteps.length - 1);
  const [copied, setCopied] = useState(false);


  const copyToClipboard = () => {
    navigator.clipboard.writeText(allSteps.join('\n'));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const colorizeCode = (line: string) => {
    // Handle empty lines
    if (line.trim() === '') {
      return <span className="text-gray-500">&nbsp;</span>;
    }

    // Handle comments
    if (line.trim().startsWith('#')) {
      return (
        <span className="text-gray-400">
          {line.replace('#', '')}
        </span>
      );
    }

    // Handle pip install command
    if (line.includes('pip install')) {
      return (
        <>
          <span className="text-blue-400">pip</span>
          <span className="text-white"> install </span>
          <span className="text-yellow-300">verbalized-sampling</span>
        </>
      );
    }

    // Handle verbalize command
    if (line.includes('verbalize run')) {
      return (
        <>
          <span className="text-blue-400">verbalize</span>
          <span className="text-white"> run </span>
          <span className="text-white">\</span>
        </>
      );
    }

    // Handle command line arguments (lines starting with spaces and --)
    if (line.trim().startsWith('--')) {
      const leadingSpaces = line.match(/^(\s*)/)?.[1] || '';
      const restOfLine = line.trim();
      // Convert spaces to non-breaking spaces to preserve them
      const preservedSpaces = leadingSpaces.replace(/ /g, '\u00A0');
      return (
        <>
          <span className="text-white">{preservedSpaces}</span>
          <span className="text-white">{restOfLine}</span>
        </>
      );
    }

    // Handle import statements
    if (line.startsWith('from ') || line.startsWith('import ')) {
      const parts = line.split(' ');
      return (
        <>
          <span className="text-blue-400">{parts[0]}</span>
          <span className="text-white"> </span>
          <span className="text-green-300">{parts.slice(1).join(' ')}</span>
        </>
      );
    }

    // Handle variable assignments
    if (line.includes(' = ')) {
      const [variable, ...rest] = line.split(' = ');
      return (
        <>
          <span className="text-yellow-300">{variable}</span>
          <span className="text-white"> = </span>
          <span className="text-blue-300">{rest.join(' = ')}</span>
        </>
      );
    }

    // Handle function calls
    if (line.includes('(') && line.includes(')')) {
      const beforeParen = line.substring(0, line.indexOf('('));
      const parenContent = line.substring(line.indexOf('('), line.lastIndexOf(')') + 1);
      const afterParen = line.substring(line.lastIndexOf(')') + 1);

      return (
        <>
          <span className="text-cyan-300">{beforeParen}</span>
          <span className="text-white">{parenContent}</span>
          <span className="text-white">{afterParen}</span>
        </>
      );
    }

    // Default case
    return <span className="text-white">{line}</span>;
  };

  return (
    <div className="w-full rounded-lg shadow-lg overflow-hidden bg-gray-900 text-white font-mono text-sm relative min-h-[200px]" style={{ height }}>
      <div className="p-4 pb-8">
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
        <div className="space-y-4 leading-tight">
          {/* Pip Install Block */}
          <div className="space-y-1">
            {pipCommand.map((step, index) => (
              <div
                key={`pip-${index}`}
                className={`${index > terminalStep ? 'opacity-0' : 'opacity-100'} transition-opacity duration-300 whitespace-nowrap`}
              >
                <span className="text-green-400">$</span> {colorizeCode(step)}
              </div>
            ))}
          </div>

          {/* Python Code Block */}
          <div className="space-y-1">
            {pythonCode.map((step, index) => {
              const globalIndex = pipCommand.length + 1 + index;
              return (
                <div
                  key={`python-${index}`}
                  className={`${globalIndex > terminalStep ? 'opacity-0' : 'opacity-100'} transition-opacity duration-300 whitespace-pre`}
                >
                  {step.trim() === '' ? (
                    colorizeCode(step)
                  ) : (
                    <>
                      <span className="text-green-400">$</span> {colorizeCode(step)}
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
