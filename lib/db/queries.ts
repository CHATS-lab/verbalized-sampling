import { desc, and, eq, isNull } from 'drizzle-orm';
// import { db } from './drizzle';
import { activityLogs, teamMembers, teams, users } from './schema';
import { cookies } from 'next/headers';
import { verifyToken } from '@/lib/auth/session';

export async function getUser() {
  return null;
}

export async function getTeamByStripeCustomerId(customerId: string) {
  return null;
}

export async function updateTeamSubscription(
  teamId: number,
  subscriptionData: {
    stripeSubscriptionId: string | null;
    stripeProductId: string | null;
    planName: string | null;
    subscriptionStatus: string;
  }
) {
  return;
}


export async function getTeamForUser() {
  return null;
}
