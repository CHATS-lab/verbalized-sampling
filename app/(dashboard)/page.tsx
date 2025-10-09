"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { ArrowRight, CreditCard, Database, Calendar, User, Clock, Eye, Heart, ThumbsUp, Share2 } from 'lucide-react';

import { Terminal } from './terminal';

// Type definitions
interface VisitorSession {
  visitorId: string;
  firstVisit: number;
  lastActivity: number;
}

interface Reactions {
  heart: number;
  thumbsUp: number;
  share: number;
}

type ReactionType = keyof Reactions;

type SharePlatform = 'twitter' | 'linkedin' | 'slack' | 'email' | 'copy';

interface ShareUrls {
  twitter: string;
  linkedin: string;
  slack: string;
  email: string;
}

interface ReactionButton {
  type: ReactionType;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  color: string;
  label: string;
}

// Reset all VS data
// localStorage.removeItem('vs_view_count');
// localStorage.removeItem('vs_reactions');
// localStorage.removeItem('vs_user_reactions');
// localStorage.removeItem('vs_visitor_sessions');
// localStorage.removeItem('vs_last_visit');
// console.log('✅ All VS data reset!');
// location.reload(); // Refresh the page


// Viewer System Component with Persistent Storage
const ViewerSystem = () => {
  const [viewCount, setViewCount] = useState<number>(0);
  const [reactions, setReactions] = useState<Reactions>({
    heart: 0,
    thumbsUp: 0,
    share: 0
  });
  const [userReactions, setUserReactions] = useState<Set<string>>(new Set());
  const [showReactionAnimation, setShowReactionAnimation] = useState<string | null>(null);
  const [showShareModal, setShowShareModal] = useState<boolean>(false);
  const [isLocalStorageAvailable, setIsLocalStorageAvailable] = useState<boolean>(true);
  
  // Generate or retrieve persistent visitor ID
  const [uniqueVisitorId] = useState<string>(() => {
    try {
      let visitorId = localStorage.getItem('vs_visitor_id');
      if (!visitorId) {
        visitorId = 'visitor_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        localStorage.setItem('vs_visitor_id', visitorId);
      }
      return visitorId;
    } catch {
      return 'visitor_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }
  });

  // Storage keys
  const STORAGE_KEYS = {
    viewCount: 'vs_view_count',
    reactions: 'vs_reactions',
    userReactions: 'vs_user_reactions',
    lastVisit: 'vs_last_visit',
    visitorSessions: 'vs_visitor_sessions',
    visitorId: 'vs_visitor_id'
  };

  // Helper function to safely use localStorage
  const safeLocalStorage = {
    getItem: (key: string): string | null => {
      try {
        return localStorage.getItem(key);
      } catch {
        return null;
      }
    },
    setItem: (key: string, value: string): boolean => {
      try {
        localStorage.setItem(key, value);
        return true;
      } catch {
        return false;
      }
    },
    removeItem: (key: string): boolean => {
      try {
        localStorage.removeItem(key);
        return true;
      } catch {
        return false;
      }
    }
  };

  // Initialize data from localStorage and track visitor
  useEffect(() => {
    // Check localStorage availability
    try {
      const testKey = '__localStorage_test__';
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
      setIsLocalStorageAvailable(true);
    } catch {
      setIsLocalStorageAvailable(false);
      console.warn('LocalStorage not available, using memory storage');
    }

    if (!isLocalStorageAvailable) {
      // Set default values for non-localStorage environment
      setViewCount(1);
      setReactions({ heart: 0, thumbsUp: 0, share: 0 });
      return;
    }

    try {
      // Load existing data or set defaults
      const savedViewCount = safeLocalStorage.getItem(STORAGE_KEYS.viewCount);
      const savedReactions = safeLocalStorage.getItem(STORAGE_KEYS.reactions);
      const savedUserReactions = safeLocalStorage.getItem(STORAGE_KEYS.userReactions);
      const savedVisitorSessions = safeLocalStorage.getItem(STORAGE_KEYS.visitorSessions);
      
      // Set reactions from storage or start at 0
      if (savedReactions) {
        const loadedReactions = JSON.parse(savedReactions);
        // Ensure all reaction types exist, even if they weren't in the saved data
        setReactions({
          heart: loadedReactions.heart || 0,
          thumbsUp: loadedReactions.thumbsUp || 0,
          share: loadedReactions.share || 0
        });
      } else {
        const initialReactions = { heart: 0, thumbsUp: 0, share: 0 };
        setReactions(initialReactions);
        safeLocalStorage.setItem(STORAGE_KEYS.reactions, JSON.stringify(initialReactions));
      }
      
      if (savedUserReactions) {
        const userReactionArray = JSON.parse(savedUserReactions);
        setUserReactions(new Set(userReactionArray));
      }

      // Track unique visitors with improved logic
      const visitorSessions: VisitorSession[] = savedVisitorSessions ? JSON.parse(savedVisitorSessions) : [];
      const now = Date.now();
      const sessionTimeout = 30 * 60 * 1000; // 30 minutes
      
      // Clean old sessions
      const activeSessions = visitorSessions.filter((session: VisitorSession) => 
        now - session.lastActivity < sessionTimeout
      );
      
      // Check if this visitor has an active session
      const existingSession = activeSessions.find((session: VisitorSession) => 
        session.visitorId === uniqueVisitorId
      );
      
      if (!existingSession) {
        // New visitor or expired session - increment view count
        activeSessions.push({
          visitorId: uniqueVisitorId,
          firstVisit: now,
          lastActivity: now
        });
        
        const currentCount = savedViewCount ? parseInt(savedViewCount) : 0;
        const newViewCount = currentCount + 1;
        setViewCount(newViewCount);
        safeLocalStorage.setItem(STORAGE_KEYS.viewCount, newViewCount.toString());
      } else {
        // Existing visitor - just update activity and load current view count
        existingSession.lastActivity = now;
        setViewCount(savedViewCount ? parseInt(savedViewCount) : 1);
      }
      
      // Save updated sessions
      safeLocalStorage.setItem(STORAGE_KEYS.visitorSessions, JSON.stringify(activeSessions));
      safeLocalStorage.setItem(STORAGE_KEYS.lastVisit, now.toString());
      
    } catch (error) {
      console.warn('Error initializing viewer data:', error);
      // Fallback to default values
      setViewCount(1);
      setReactions({ heart: 0, thumbsUp: 0, share: 0 });
    }
  }, [uniqueVisitorId, isLocalStorageAvailable]);

  // Periodic updates - simulate real-time activity (reduced frequency and better logic)
  useEffect(() => {
    if (!isLocalStorageAvailable) return;

    const interval = setInterval(() => {
      try {
        // Only simulate occasional new visitors, not frequent ones
        if (Math.random() < 0.05) { // 5% chance every 30 seconds (reduced from 20% every 10 seconds)
          const savedViewCount = safeLocalStorage.getItem(STORAGE_KEYS.viewCount);
          const currentCount = savedViewCount ? parseInt(savedViewCount) : 0;
          const increment = Math.floor(Math.random() * 2) + 1; // 1-2 new views
          const newViewCount = currentCount + increment;
          
          setViewCount(newViewCount);
          safeLocalStorage.setItem(STORAGE_KEYS.viewCount, newViewCount.toString());
          
          // Add simulated visitor sessions
          const savedVisitorSessions = safeLocalStorage.getItem(STORAGE_KEYS.visitorSessions);
          const visitorSessions: VisitorSession[] = savedVisitorSessions ? JSON.parse(savedVisitorSessions) : [];
          const now = Date.now();
          const sessionTimeout = 30 * 60 * 1000;
          
          // Clean old sessions first
          const activeSessions = visitorSessions.filter((session: VisitorSession) => 
            now - session.lastActivity < sessionTimeout
          );
          
          // Add new simulated sessions
          for (let i = 0; i < increment; i++) {
            activeSessions.push({
              visitorId: 'simulated_' + Math.random().toString(36).substr(2, 9) + '_' + now,
              firstVisit: now,
              lastActivity: now
            });
          }
          
          safeLocalStorage.setItem(STORAGE_KEYS.visitorSessions, JSON.stringify(activeSessions));
        }
      } catch (error) {
        console.warn('Error updating visitor data:', error);
      }
    }, 30000); // Check every 30 seconds instead of 10

    return () => clearInterval(interval);
  }, [isLocalStorageAvailable]);

  // Update last activity periodically
  useEffect(() => {
    const activityInterval = setInterval(() => {
      try {
        const savedVisitorSessions = localStorage.getItem(STORAGE_KEYS.visitorSessions);
        if (savedVisitorSessions) {
          const visitorSessions: VisitorSession[] = JSON.parse(savedVisitorSessions);
          const now = Date.now();
          
          const updatedSessions = visitorSessions.map((session: VisitorSession) => 
            session.visitorId === uniqueVisitorId 
              ? { ...session, lastActivity: now }
              : session
          );
          
          localStorage.setItem(STORAGE_KEYS.visitorSessions, JSON.stringify(updatedSessions));
        }
      } catch (error) {
        console.warn('Error updating activity:', error);
      }
    }, 60000); // Update every minute

    return () => clearInterval(activityInterval);
  }, [uniqueVisitorId]);

  const handleReaction = (reactionType: string) => {
    // Special handling for share - always increment and show social media options
    if (reactionType === 'share') {
      handleShare();
      return;
    }

    const newUserReactions = new Set(userReactions);
    
    try {
      if (userReactions.has(reactionType)) {
        // Remove reaction
        newUserReactions.delete(reactionType);
        setReactions(prev => {
          const updated = {
            ...prev,
            [reactionType as keyof Reactions]: Math.max(0, prev[reactionType as keyof Reactions] - 1)
          };
          localStorage.setItem(STORAGE_KEYS.reactions, JSON.stringify(updated));
          return updated;
        });
      } else {
        // Add reaction
        newUserReactions.add(reactionType);
        setReactions(prev => {
          const updated = {
            ...prev,
            [reactionType as keyof Reactions]: prev[reactionType as keyof Reactions] + 1
          };
          localStorage.setItem(STORAGE_KEYS.reactions, JSON.stringify(updated));
          return updated;
        });
        
        // Show animation
        setShowReactionAnimation(reactionType);
        setTimeout(() => setShowReactionAnimation(null), 500);
      }
      
      setUserReactions(newUserReactions);
      localStorage.setItem(STORAGE_KEYS.userReactions, JSON.stringify([...newUserReactions]));
      
    } catch (error) {
      console.warn('Error saving reaction data:', error);
    }
  };

  const handleShare = () => {
    // Increment share count
    setReactions(prev => {
      const updated = {
        ...prev,
        share: prev.share + 1
      };
      try {
        localStorage.setItem(STORAGE_KEYS.reactions, JSON.stringify(updated));
      } catch (error) {
        console.warn('Error saving share data:', error);
      }
      return updated;
    });

    // Show animation
    setShowReactionAnimation('share');
    setTimeout(() => setShowReactionAnimation(null), 500);

    // Create share data
    const shareData = {
      title: 'CollabLLM: From Passive Responders to Active Collaborators',
      text: 'Sharing the blog: "Building the Future of Collaborative AI: Our Journey with CollabLLM" - A unified fine-tuning framework that optimizes LLMs for effective multiturn collaboration.',
      url: `${window.location.origin}${window.location.pathname}#blog`
    };

    // Try native share API first (mobile devices)
    if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
      navigator.share(shareData).catch(() => {
        // Fallback to modal if native share fails
        setShowShareModal(true);
      });
    } else {
      // Desktop fallback - show share modal
      setShowShareModal(true);
    }
  };

  const shareToSocial = (platform: SharePlatform) => {
    const shareData = {
      title: 'CollabLLM: From Passive Responders to Active Collaborators',
      text: 'Sharing the blog: "Building the Future of Collaborative AI: Our Journey with CollabLLM" - A unified fine-tuning framework that optimizes LLMs for effective multiturn collaboration.',
      url: `${window.location.origin}${window.location.pathname}#blog`
    };

    const shareUrls: ShareUrls = {
      twitter: `https://twitter.com/messages/compose?text=${encodeURIComponent(`${shareData.text} ${shareData.url}`)}`,
      linkedin: `https://www.linkedin.com/messaging/`,
      slack: `slack://open`,
      email: `mailto:?subject=${encodeURIComponent(shareData.title)}&body=${encodeURIComponent(`${shareData.text}\n\n${shareData.url}`)}`
    };

    if (platform === 'copy') {
      // Copy link to clipboard
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(shareData.url).then(() => {
          alert('✅ Link copied to clipboard!');
          setShowShareModal(false);
        }).catch(() => {
          // Fallback for clipboard API failure
          const textArea = document.createElement('textarea');
          textArea.value = shareData.url;
          document.body.appendChild(textArea);
          textArea.select();
          try {
            document.execCommand('copy');
            alert('✅ Link copied to clipboard!');
            setShowShareModal(false);
          } catch (err) {
            prompt('Copy this link manually:', shareData.url);
          }
          document.body.removeChild(textArea);
        });
      } else {
        // Old browser fallback
        prompt('Copy this link:', shareData.url);
        setShowShareModal(false);
      }
    } else if (platform === 'linkedin') {
      // Special handling for LinkedIn DM - copy message and open LinkedIn
      const linkedinMessage = `${shareData.text}\n\n${shareData.url}`;
      
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(linkedinMessage).then(() => {
          alert('✅ Message copied to clipboard! Opening LinkedIn...\n\nPaste the message in your DM.');
          window.open(shareUrls.linkedin, '_blank', 'width=800,height=600,scrollbars=yes,resizable=yes');
          setShowShareModal(false);
        }).catch(() => {
          // Fallback
          prompt('Copy this message for LinkedIn DM:', linkedinMessage);
          window.open(shareUrls.linkedin, '_blank', 'width=800,height=600,scrollbars=yes,resizable=yes');
          setShowShareModal(false);
        });
      } else {
        // Old browser fallback
        prompt('Copy this message for LinkedIn DM:', linkedinMessage);
        window.open(shareUrls.linkedin, '_blank', 'width=800,height=600,scrollbars=yes,resizable=yes');
        setShowShareModal(false);
      }
    } else if (platform === 'slack') {
      // Special handling for Slack - copy message and open Slack
      const slackMessage = `${shareData.text}\n\n${shareData.url}`;
      
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(slackMessage).then(() => {
          alert('✅ Message copied to clipboard! Opening Slack...\n\nPaste the message in any channel or DM.');
          window.open(shareUrls.slack, '_blank');
          setShowShareModal(false);
        }).catch(() => {
          // Fallback
          prompt('Copy this message for Slack:', slackMessage);
          window.open(shareUrls.slack, '_blank');
          setShowShareModal(false);
        });
      } else {
        // Old browser fallback
        prompt('Copy this message for Slack:', slackMessage);
        window.open(shareUrls.slack, '_blank');
        setShowShareModal(false);
      }
    } else {
      // Open other social media share URLs
      const newWindow = window.open(shareUrls[platform as keyof ShareUrls], '_blank', 'width=600,height=400,scrollbars=yes,resizable=yes');
      if (!newWindow) {
        alert('Pop-up blocked! Please allow pop-ups for sharing or copy the link manually.');
      }
      setShowShareModal(false);
    }
  };

  // Share Modal Component
  const ShareModal = () => {
    if (!showShareModal) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowShareModal(false)}>
        <div className="bg-white rounded-lg p-6 max-w-sm w-full mx-4 shadow-lg" onClick={e => e.stopPropagation()}>
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Share this article</h3>
          
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => shareToSocial('twitter')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors"
            >
              <div className="w-5 h-5 bg-blue-400 rounded"></div>
              <span className="text-sm font-medium">Twitter DM</span>
            </button>
            
            <button
              onClick={() => shareToSocial('linkedin')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors"
            >
              <div className="w-5 h-5 bg-blue-600 rounded"></div>
              <span className="text-sm font-medium">LinkedIn DM</span>
            </button>
            
            <button
              onClick={() => shareToSocial('slack')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-purple-50 hover:border-purple-300 transition-colors"
            >
              <div className="w-5 h-5 bg-purple-500 rounded"></div>
              <span className="text-sm font-medium">Slack</span>
            </button>
            
            <button
              onClick={() => shareToSocial('email')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-gray-300 transition-colors"
            >
              <div className="w-5 h-5 bg-gray-500 rounded"></div>
              <span className="text-sm font-medium">Email</span>
            </button>
            
            <button
              onClick={() => shareToSocial('copy')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-green-50 hover:border-green-300 transition-colors col-span-2"
            >
              <div className="w-5 h-5 bg-green-500 rounded"></div>
              <span className="text-sm font-medium">Copy Link</span>
            </button>
          </div>
          
          <button
            onClick={() => setShowShareModal(false)}
            className="w-full mt-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  };

  const reactionButtons: ReactionButton[] = [
    { type: 'heart', icon: Heart, color: 'text-red-500', label: 'Love' },
    { type: 'thumbsUp', icon: ThumbsUp, color: 'text-blue-500', label: 'Good' },
    { type: 'share', icon: Share2, color: 'text-green-500', label: 'Share' }
  ];

  return (
    <>
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 py-3">
        {/* Share Button Only */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleReaction('share')}
            className="flex items-center gap-2 px-4 py-2 rounded-full border border-gray-300 hover:border-gray-400 text-gray-500 hover:text-green-500 transition-all duration-200 hover:scale-105"
            title="Share this article"
          >
            <Share2 size={16} />
            <span className="text-sm font-medium">Share</span>
          </button>
        </div>
      </div>
      {/* Share Modal */}
      <ShareModal />
    </>
  );
};

export default function HomePage() {
  return (
    <main>
      <section className="py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Paper Title and Authors Section */}
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-700 tracking-tight sm:text-5xl mb-6">
              Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity
            </h1>
            
            <div className="text-xl text-gray-600 mb-2 max-w-5xl mx-auto leading-relaxed">
              <div className="mb-1">
                <a href="https://jiayizx.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Jiayi Zhang</a><sup className="text-orange-500">1</sup><sup className="text-orange-500">†</sup>,{' '}
                <a href="https://simonucl.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Simon Yu</a><sup className="text-orange-500">1</sup><sup className="text-orange-500">†</sup>,{' '}
                <a href="https://www.linkedin.com/in/derekch" className="text-blue-400 hover:text-blue-400 transition-colors">Derek Chong</a><sup className="text-orange-500">2</sup><sup className="text-orange-500">†</sup>,{' '}
                <a href="https://anthonysicilia.tech/" className="text-blue-400 hover:text-blue-400 transition-colors">Anthony Sicilia</a><sup className="text-orange-500">3</sup>
              </div>
              <div>
                <a href="https://tomz.people.stanford.edu/" className="text-blue-400 hover:text-blue-400 transition-colors">Michael R. Tomz</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://nlp.stanford.edu/~manning/" className="text-blue-400 hover:text-blue-400 transition-colors">Christopher D. Manning</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://wyshi.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Weiyan Shi</a><sup className="text-orange-500">1</sup>
              </div>
            </div>
            
            <div className="text-xl text-black-500 mb-4">
              <sup className="text-orange-500">1</sup>Northeastern University, <sup className="text-orange-500">2</sup>Stanford University, <sup className="text-orange-500">3</sup>West Virginia University<br/>
              <sup className="text-orange-500">†</sup>Equal contribution
            </div>
            
            {/* <div className="text-xl font-bold">
              <span className="text-orange-600">ICML 2025 Outstanding Paper</span>
            </div> */}
          </div>
        </div>
      </section>
      
      {/* Introduction */}
      <section className="py-8 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <img 
              src="/images/intro_teaser.jpg" 
              alt="Verbalized Sampling Introduction"
              className="w-full max-w-5xl mx-auto rounded-lg shadow-lg"
            />
            <div className="text-sm text-gray-500 mt-2">
              <strong>Figure 1:</strong> Overview of Verbalized Sampling (VS) for unlocking LLM diversity.
            </div>
          </div>
        </div>
      </section>

      <section className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-8">
              Abstract
            </h2>
            <div className="max-w-5xl mx-auto">
              <p className="text-lg text-gray-600 leading-relaxed text-left">
                Post-training alignment often reduces LLM diversity, leading to a phenomenon known as <em>mode collapse</em>. 
                Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: <em>typicality bias</em> in preference data, 
                whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. 
                We formalize this bias theoretically, verify it empirically on preference datasets, and show that it plays a central role in mode collapse. 
              </p>
              <p className="text-lg text-gray-600 leading-relaxed text-left mt-6">
                Motivated by this analysis, we introduce <strong>Verbalized Sampling (VS)</strong>, a simple, training-free prompting method to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities").
                Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1× over direct prompting. We further observe an emergent trend that more capable models benefit more from VS.
                In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
              </p>
            </div>
          </div>
        </div>
        </section>

        <section className="py-12 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="lg:grid lg:grid-cols-2 lg:gap-12 lg:items-center">
              <div>
                  <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-6">
                    Data-Level Cause of<br />
                    Mode Collapse: <span className="text-orange-500">Typicality Bias</span>
                  </h2>
                 <div className="space-y-4 text-lg text-gray-600">
                   <p>
                     Cognitive psychology shows that people prefer text that is familiar, fluent, and predictable.
                     We use base models as human proxies and verify this on multiple preference datasets and base models, 
                     confirming that the typicality bias exists (see Figure 2).
                   </p>
                     <p>
                       This bias sharpens the probability distribution towards a few stereotypical completions during RLHF stages.
                       When many high-quality completions are possible (e.g., in story generation), this sharpening becomes
                       a tie-breaker, resulting in mode collapse.
                     </p>
                 </div>
              </div>
              <div className="mt-8 lg:mt-0">
                <img 
                  src="/images/cognitive_bias_combined.jpg" 
                  alt="Cognitive Bias and Typicality in Preference Data"
                  className="w-full rounded-lg shadow-lg"
                />
                <p className="text-sm text-gray-500 mt-3 text-center italic">
                  <strong>Figure 2:</strong> How often the human-preferred response in a preference pair is assigned a higher log likelihood by a base model.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-12 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="lg:grid lg:grid-cols-2 lg:gap-12 lg:items-center">
              <div className="mt-8 lg:mt-0">
                <img 
                  src="/images/prompting_methods.png" 
                  alt="Prompting Methods Comparison"
                  className="w-full rounded-lg shadow-lg"
                />
                <p className="text-sm text-gray-500 mt-3 text-center italic">
                  <strong>Figure 3:</strong> Three types of prompting methods: instance-level, list-level, and distribution-level, given the same computation budget of N total responses.
                </p>
              </div>
              <div>
                <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-6">
                  Three Types of
                  <span className="block text-orange-500">Prompting Methods</span>
                </h2>
                  <div className="space-y-4 text-lg text-gray-600">
                    <p>
                      Motivated by the theoretical understanding of mode collapse, we propose Verbalized Sampling (VS) and formalize prompting methods into three categories, each with their corresponding modes (see Figure 3):
                    </p>
                    <p>
                      <strong>Instance-level prompt:</strong> The most traditional prompt requesting one instance (e.g., "Tell me a joke about coffee"). 
                      The mode is the mode instance of the base model.
                    </p>
                    <p>
                      <strong>List-level prompt:</strong> Requests a list of outputs (e.g., "Tell me k jokes about coffee"). 
                      The mode is a uniform distribution of related items learned by the base model during pretraining.
                    </p>
                    <p>
                      <strong>Distribution-level prompt (Verbalized Sampling):</strong> Requests k outputs with corresponding probabilities 
                      (e.g., "Tell k jokes about coffee with their probabilities"). The mode is a distribution capable of approximating 
                      the distribution of related items learned by the base model during pretraining.
                    </p>
                  </div>
              </div>
            </div>
          </div>
        </section>

        <section className="py-12 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="max-w-5xl mx-auto text-center">
              <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-6">
                Verbalized Sampling Works On
                <span className="block text-orange-500">Multiple Tasks</span>
              </h2>
              <div className="text-center mb-8">
                <img 
                  src="/images/qualitative_examples.png" 
                  alt="Qualitative Examples Across Multiple Tasks"
                  className="w-full max-w-6xl mx-auto rounded-lg shadow-lg"
                />
                <p className="text-sm text-gray-500 mt-3 text-center italic">
                  <strong>Figure 4:</strong> Qualitative and quantitative examples of Verbalized Sampling on creative writing, dialogue simulation, and enumerative open-ended QA.
                </p>
              </div>
               <div className="space-y-4 text-lg text-gray-600 text-left">
                 <p>
                   Our comprehensive experiments on multiple tasks demonstrate that Verbalized Sampling significantly improves the diversity-quality trade-off across tasks and model families, 
                   without compromising factual accuracy and safety.
                 </p>
                 <p>
                   As shown in Figure 4, for <strong>story writing</strong>, VS improves the output diversity. 
                   For <strong>dialogue simulation</strong>, VS simulates the donation amount distribution much closer to the human distribution, and generates more realistic persuasion behaviors.
                   On the task of <strong>enumerative open-ended QA</strong>, we ask the model to "generate US states". We first query a pretraining corpus (RedPajama) to establish a "reference" distribution of US 
                   state names in the pretraining data. The verbalized probability distribution generated by VS, when averaged over 10 trials, closely aligns with this reference pretraining distribution (KL=0.12). 
                   In contrast, direct prompting collapses into a few modes, repeatedly outputting states like California and Texas. 
                 </p>
                 <div className="lg:grid lg:grid-cols-2 lg:gap-12 lg:items-center">
                   <div className="space-y-4 text-lg text-gray-600 text-left">
                     <p>
                       We observe an <strong>emergent trend</strong> where larger models benefit more from VS. Figure 5 shows the diversity gain over the direct prompting which suffers from mode collapse. 
                       Across all VS variants, larger models (GPT-4.1, Gemini-2.5-Pro) achieve diversity gains 1.5 to 2 times greater than smaller models (GPT-4.1-Mini, Gemini-2.5-Flash).
                     </p>
                   </div>
                   <div className="mt-8 lg:mt-0">
                     <img 
                       src="/images/emergent_trend.png" 
                       alt="Emergent Trend: Larger Models Benefit More from VS"
                       className="w-full rounded-lg shadow-lg"
                     />
                     <p className="text-sm text-gray-500 mt-3 text-center italic">
                       <strong>Figure 5:</strong> Emergent trend where larger models benefit more from VS. We show differences in
                       diversity (e) and quality (f) over Direct across small and large models.
                     </p>
                   </div>
                 </div>
               </div>
            </div>
          </div>
        </section>

        <section className="py-12 bg-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="lg:grid lg:grid-cols-2 lg:gap-12 lg:items-center">
              <div>
                <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-6">
                  Pro-tip for
                  <span className="block text-orange-500">Maximizing Diversity</span>
                </h2>
                <div className="space-y-4 text-lg text-gray-600">
                  <p>
                    Unlike baseline methods, Verbalized Sampling allows us to tune the output diversity by adjusting
                    the probability threshold directly in the prompt (e.g., "Generate five responses with probabilities
                    below &lt;threshold&gt;"), without altering decoding parameters. As shown in Figure 6, diversity
                    increases as the probability threshold decreases.
                  </p>
                </div>
              </div>
              <div className="mt-8 lg:mt-0">
                <img 
                  src="/images/prob_tuning.png" 
                  alt="Probability Tuning for Maximum Diversity"
                  className="w-full rounded-lg shadow-lg"
                />
                <p className="text-sm text-gray-500 mt-3 text-center italic">
                  <strong>Figure 6:</strong> Tunable Diversity shows the diversity tuning results on Gemini-2.5-Flash across tasks. 
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-12 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-12 lg:gap-8">
            <div className="sm:text-center md:max-w-2xl md:mx-auto lg:col-span-7 lg:text-left">
              <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-4">
                Try the Magic Prompt:
                <span className="block text-orange-500">Unlock LLM Diversity</span>
              </h2>
              <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-xl lg:text-lg xl:text-xl">
              Verbalized Sampling provides a training-free, model-agnostic approach to mitigating mode collapse by prompting the model to generate response distributions with verbalized probability estimates.
              </p>
              {/* <div className="mt-8 sm:max-w-lg sm:mx-auto sm:text-center lg:text-left lg:mx-0">
                <a
                  href="https://huggingface.co/verbalized-sampling"
                  target="_blank"
                >
                  <Button
                    size="lg"
                    variant="outline"
                    className="text-lg rounded-full"
                  >
                    Try Verbalized Sampling
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                </a>
              </div> */}
            </div>
            <div className="mt-12 relative sm:max-w-lg sm:mx-auto lg:mt-0 lg:max-w-none lg:mx-0 lg:col-span-5 lg:flex lg:items-center">
              <Terminal />
            </div>
          </div>
        </div>
      </section>


      {/* BibTeX Citation */}
      <section className="py-12 bg-white w-full">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-700 sm:text-4xl mb-8">
              📌 BibTeX Citation
            </h2>
            <p className="text-lg text-gray-600 mb-8">
              If you find our project useful, please consider citing:
            </p>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 w-full max-w-6xl mx-auto">
              <pre className="text-sm text-gray-800 font-mono whitespace-pre-wrap text-left overflow-x-auto">
{`@misc{zhang2025verbalizedsamplingmitigatemode,
      title={Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity}, 
      author={Jiayi Zhang and Simon Yu and Derek Chong and Anthony Sicilia and Michael R. Tomz and Christopher D. Manning and Weiyan Shi},
      year={2025},
      eprint={2510.01171},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.01171}, 
}`}
              </pre>
            </div>
          </div>
        </div>
      </section>


    </main>
  );
}