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

// Reset all CollabLLM data
// localStorage.removeItem('collabllm_view_count');
// localStorage.removeItem('collabllm_reactions');
// localStorage.removeItem('collabllm_user_reactions');
// localStorage.removeItem('collabllm_visitor_sessions');
// localStorage.removeItem('collabllm_last_visit');
// console.log('✅ All CollabLLM data reset!');
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
      let visitorId = localStorage.getItem('collabllm_visitor_id');
      if (!visitorId) {
        visitorId = 'visitor_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        localStorage.setItem('collabllm_visitor_id', visitorId);
      }
      return visitorId;
    } catch {
      return 'visitor_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }
  });

  // Storage keys
  const STORAGE_KEYS = {
    viewCount: 'collabllm_view_count',
    reactions: 'collabllm_reactions',
    userReactions: 'collabllm_user_reactions',
    lastVisit: 'collabllm_last_visit',
    visitorSessions: 'collabllm_visitor_sessions',
    visitorId: 'collabllm_visitor_id'
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
                <a href="https://jiayizx.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Jiayi Zhang</a><sup className="text-orange-500">1</sup>,{' '}
                <a href="https://simonucl.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Simon Yu</a><sup className="text-orange-500">1</sup>,{' '}
                <a href="https://www.linkedin.com/in/derekch" className="text-blue-400 hover:text-blue-400 transition-colors">Derek Chong</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://anthonysicilia.tech/" className="text-blue-400 hover:text-blue-400 transition-colors">Anthony Sicilia</a><sup className="text-orange-500">3</sup>
              </div>
              <div>
                <a href="https://tomz.people.stanford.edu/" className="text-blue-400 hover:text-blue-400 transition-colors">Michael R. Tomz</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://nlp.stanford.edu/~manning/" className="text-blue-400 hover:text-blue-400 transition-colors">Christopher D. Manning</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://wyshi.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Weiyan Shi</a><sup className="text-orange-500">1</sup>
              </div>
            </div>
            
            <div className="text-xl text-black-500 mb-4">
              <sup className="text-orange-500">1</sup>Northeastern University, <sup className="text-orange-500">2</sup>Stanford University, <sup className="text-orange-500">3</sup>West Virginia University
            </div>
            
            {/* <div className="text-xl font-bold">
              <span className="text-orange-600">ICML 2025 Outstanding Paper</span>
            </div> */}
          </div>
        </div>
      </section>
      

      <section className="py-8  bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-12 lg:gap-8">
            <div className="sm:text-center md:max-w-2xl md:mx-auto lg:col-span-7 lg:text-left">
              <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-4">
                Make Your LLMs
                <span className="block text-orange-500">Active Collaborators</span>
              </h2>
              <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-xl lg:text-lg xl:text-xl">
              CollabLLM is a unified fine-tuning framework that optimizes LLMs for 
              effective and efficient multiturn collaboration with users.
              </p>
              <div className="mt-8 sm:max-w-lg sm:mx-auto sm:text-center lg:text-left lg:mx-0">
                <a
                  href="https://huggingface.co/collabllm"
                  target="_blank"
                >
                  <Button
                    size="lg"
                    variant="outline"
                    className="text-lg rounded-full"
                  >
                    Try CollabLLM Models
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                </a>
              </div>
            </div>
            <div className="mt-12 relative sm:max-w-lg sm:mx-auto lg:mt-0 lg:max-w-none lg:mx-0 lg:col-span-5 lg:flex lg:items-center">
              <Terminal />
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-gray-50 w-full">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-3 lg:gap-8">
            <div>
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-orange-500 text-white">
                <svg viewBox="0 0 24 24" className="h-6 w-6">
                  <path
                    fill="currentColor"
                    d="M14.23 12.004a2.236 2.236 0 0 1-2.235 2.236 2.236 2.236 0 0 1-2.236-2.236 2.236 2.236 0 0 1 2.235-2.236 2.236 2.236 0 0 1 2.236 2.236zm2.648-10.69c-1.346 0-3.107.96-4.888 2.622-1.78-1.653-3.542-2.602-4.887-2.602-.41 0-.783.093-1.106.278-1.375.793-1.683 3.264-.973 6.365C1.98 8.917 0 10.42 0 12.004c0 1.59 1.99 3.097 5.043 4.03-.704 3.113-.39 5.588.988 6.38.32.187.69.275 1.102.275 1.345 0 3.107-.96 4.888-2.624 1.78 1.654 3.542 2.603 4.887 2.603.41 0 .783-.09 1.106-.275 1.374-.792 1.683-3.263.973-6.365C22.02 15.096 24 13.59 24 12.004c0-1.59-1.99-3.097-5.043-4.032.704-3.11.39-5.587-.988-6.38-.318-.184-.688-.277-1.092-.278zm-.005 1.09v.006c.225 0 .406.044.558.127.666.382.955 1.835.73 3.704-.054.46-.142.945-.25 1.44-.96-.236-2.006-.417-3.107-.534-.66-.905-1.345-1.727-2.035-2.447 1.592-1.48 3.087-2.292 4.105-2.295zm-9.77.02c1.012 0 2.514.808 4.11 2.28-.686.72-1.37 1.537-2.02 2.442-1.107.117-2.154.298-3.113.538-.112-.49-.195-.964-.254-1.42-.23-1.868.054-3.32.714-3.707.19-.09.4-.127.563-.132zm4.882 3.05c.455.468.91.992 1.36 1.564-.44-.02-.89-.034-1.345-.034-.46 0-.915.01-1.36.034.44-.572.895-1.096 1.345-1.565zM12 8.1c.74 0 1.477.034 2.202.093.406.582.802 1.203 1.183 1.86.372.64.71 1.29 1.018 1.946-.308.655-.646 1.31-1.013 1.95-.38.66-.773 1.288-1.18 1.87-.728.063-1.466.098-2.21.098-.74 0-1.477-.035-2.202-.093-.406-.582-.802-1.204-1.183-1.86-.372-.64-.71-1.29-1.018-1.946.303-.657.646-1.313 1.013-1.954.38-.66.773-1.286 1.18-1.868.728-.064 1.466-.098 2.21-.098zm-3.635.254c-.24.377-.48.763-.704 1.16-.225.39-.435.782-.635 1.174-.265-.656-.49-1.31-.676-1.947.64-.15 1.315-.283 2.015-.386zm7.26 0c.695.103 1.365.23 2.006.387-.18.632-.405 1.282-.66 1.933-.2-.39-.41-.783-.64-1.174-.225-.392-.465-.774-.705-1.146zm3.063.675c.484.15.944.317 1.375.498 1.732.74 2.852 1.708 2.852 2.476-.005.768-1.125 1.74-2.857 2.475-.42.18-.88.342-1.355.493-.28-.958-.646-1.956-1.1-2.98.45-1.017.81-2.01 1.085-2.964zm-13.395.004c.278.96.645 1.957 1.1 2.98-.45 1.017-.812 2.01-1.086 2.964-.484-.15-.944-.318-1.37-.5-1.732-.737-2.852-1.706-2.852-2.474 0-.768 1.12-1.742 2.852-2.476.42-.18.88-.342 1.356-.494zm11.678 4.28c.265.657.49 1.312.676 1.948-.64.157-1.316.29-2.016.39.24-.375.48-.762.705-1.158.225-.39.435-.788.636-1.18zm-9.945.02c.2.392.41.783.64 1.175.23.39.465.772.705 1.143-.695-.102-1.365-.23-2.006-.386.18-.63.406-1.282.66-1.933zM17.92 16.32c.112.493.2.968.254 1.423.23 1.868-.054 3.32-.714 3.708-.147.09-.338.128-.563.128-1.012 0-2.514-.807-4.11-2.28.686-.72 1.37-1.536 2.02-2.44 1.107-.118 2.154-.3 3.113-.54zm-11.83.01c.96.234 2.006.415 3.107.532.66.905 1.345 1.727 2.035 2.446-1.595 1.483-3.092 2.295-4.11 2.295-.22-.005-.406-.05-.553-.132-.666-.38-.955-1.834-.73-3.703.054-.46.142-.944.25-1.438zm4.56.64c.44.02.89.034 1.345.034.46 0 .915-.01 1.36-.034-.44.572-.895 1.095-1.345 1.565-.455-.47-.91-.993-1.36-1.565z"
                  />
                </svg>
              </div>
              <div className="mt-5">
                <h2 className="text-lg font-medium text-gray-700">
                  What is missing from current LLMs?
                </h2>
                <p className="mt-2 text-base text-gray-500">
                  LLMs act as passive responders, especially when faced with ambiguous inputs. They don't naturally help users explore their needs in multiturn interations or offer suggestions for next steps.
                </p>
              </div>
            </div>

            <div className="mt-10 lg:mt-0">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-orange-500 text-white">
                <Database className="h-6 w-6" />
              </div>
              <div className="mt-5">
                <h2 className="text-lg font-medium text-gray-700">
                  Why do LLMs fail to understand users?
                </h2>
                <p className="mt-2 text-base text-gray-500">
                Most LLMs are tuned based on single-turn human preferences. These single-turn rewards encourage models to generate response that may NOT be useful in the long term.
                </p>
              </div>
            </div>

            <div className="mt-10 lg:mt-0">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-orange-500 text-white">
                <CreditCard className="h-6 w-6" />
              </div>
              <div className="mt-5">
                <h2 className="text-lg font-medium text-gray-700">
                  How do we build collaborative LLMs?
                </h2>
                <p className="mt-2 text-base text-gray-500">
                CollabLLM rewards LLMs responses based on their long-term impact on the conversation. By finetune LLMs using these long-term, interaction-level rewards, they actively seek information and collaborate more effectively with users.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-8 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-left">
            <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-8">
              What Users Said
              <span className=" text-orange-500"> About CollabLLM</span>
            </h2>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
              {/* Quote 1 */}
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="text-orange-500 text-2xl mb-4">&ldquo;Efficient</div>
                <p className="text-gray-600 italic mb-4">
                I was surprised by the first response. I was expecting a quick summary related to my prompt, but instead the AI asked me some questions. 
                I think this style worked well. 
                {/* It really helped me get detailed writing from the start.  */}
                {/* The response after that was more nuanced.  */}
                I felt like I had to do <strong>less editing</strong> to personalize the review.
                </p>
              </div>
              {/* Quote 2 */}
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="text-orange-500 text-2xl mb-4">&ldquo;Stimulate Creativity</div>
                <p className="text-gray-600 italic mb-4">
                Asking questions and making you think of things you <strong>never thought of</strong>.
                </p>

                {/* <p className="text-gray-600 italic mb-4">
                Had some <strong>interesting ideas</strong> and asked good questions.
                </p> */}

                <p className="text-gray-600 italic mb-4">
                The AI assistant listened extremely well and offered suggestions that made sense as if it were a <strong>real conversation</strong>
                </p>                
              </div>

              {/* Quote 3 */}
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="text-orange-500 text-2xl mb-4">&ldquo;Safer</div>
                <p className="text-gray-600 italic mb-4">
                The AI assistant told me why it <strong>wouldn't be helpful</strong> for this case.
                </p>
                <p className="text-gray-600 italic mb-4">
                It helped really well to navigate what to say and <strong>what information is needed</strong>.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-12 bg-gray-50 w-full">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-2 lg:gap-8 lg:items-center">
            <div>
              <h2 className="text-3xl font-bold text-gray-700 sm:text-4xl">
                Ready to Make Your LLMs <span className=" text-orange-500"> Collaborative?</span>
              </h2>
              <p className="mt-3 max-w-3xl text-lg text-gray-500">
                Our code makes it easy for you to get more collaborative LLMs on your own tasks. 
                Don't waste time interacting with LLMs that fail to understand your need, start building collaborative LLMs!
              </p>
            </div>
            <div className="mt-8 lg:mt-0 flex justify-center lg:justify-end">
              <a href="https://github.com/Wuyxin/collabllm.git" target="_blank">
                <Button
                  size="lg"
                  variant="outline"
                  className="text-lg rounded-full"
                >
                  View code on Github
                  <ArrowRight className="ml-3 h-6 w-6" />
                </Button>
              </a>
            </div>
          </div>
        </div>
      </section>


    </main>
  );
}