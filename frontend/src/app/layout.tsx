import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Aviation Document AI Chat",
  description: "Ask questions about aviation documents with AI-powered retrieval and grounded answers. Built with hybrid RAG (BM25 + Vector + Reranker).",
  keywords: ["aviation", "RAG", "AI", "chat", "documents", "ATPL", "PPL"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}

