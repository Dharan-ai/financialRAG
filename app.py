"""
Production-Ready Financial RAG System using LangGraph
Analyzes bank statements to generate P&L, Balance Sheet, and other financial reports
"""

import os
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import json
from datetime import datetime
import operator

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL_NAME = "claude-sonnet-4-20250514"

class FinancialState(TypedDict):
    """State for the financial analysis workflow"""
    uploaded_file: str
    extracted_transactions: List[dict]
    categorized_transactions: List[dict]
    profit_loss: dict
    balance_sheet: dict
    cash_flow: dict
    analysis_summary: str
    error: str
    messages: Annotated[List[str], operator.add]

class FinancialRAGSystem:
    """LangGraph-based RAG system for financial analysis"""
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model=MODEL_NAME,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0
        )
        
        # Initialize embeddings (using free HuggingFace model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Build the graph
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(FinancialState)
        
        # Add nodes
        workflow.add_node("extract_transactions", self.extract_transactions)
        workflow.add_node("categorize_transactions", self.categorize_transactions)
        workflow.add_node("create_vector_store", self.create_vector_store)
        workflow.add_node("calculate_profit_loss", self.calculate_profit_loss)
        workflow.add_node("calculate_balance_sheet", self.calculate_balance_sheet)
        workflow.add_node("calculate_cash_flow", self.calculate_cash_flow)
        workflow.add_node("generate_analysis", self.generate_analysis)
        
        # Define edges
        workflow.set_entry_point("extract_transactions")
        workflow.add_edge("extract_transactions", "categorize_transactions")
        workflow.add_edge("categorize_transactions", "create_vector_store")
        workflow.add_edge("create_vector_store", "calculate_profit_loss")
        workflow.add_edge("calculate_profit_loss", "calculate_balance_sheet")
        workflow.add_edge("calculate_balance_sheet", "calculate_cash_flow")
        workflow.add_edge("calculate_cash_flow", "generate_analysis")
        workflow.add_edge("generate_analysis", END)
        
        return workflow
    
    def extract_transactions(self, state: FinancialState) -> dict:
        """Extract transactions from uploaded bank statement"""
        try:
            file_path = state.get("uploaded_file", "")
            
            if not file_path or not os.path.exists(file_path):
                return {"error": "File not found", "messages": ["Error: File not found"]}
            
            # Read the file based on extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return {"error": "Unsupported file format", "messages": ["Error: Unsupported file format"]}
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Extract transactions
            transactions = []
            for _, row in df.iterrows():
                transaction = {
                    'date': self._parse_date(row.get('date', row.get('transaction date', ''))),
                    'description': str(row.get('description', row.get('narration', row.get('particulars', '')))),
                    'debit': float(row.get('debit', row.get('withdrawal', 0)) or 0),
                    'credit': float(row.get('credit', row.get('deposit', 0)) or 0),
                    'balance': float(row.get('balance', 0) or 0)
                }
                transactions.append(transaction)
            
            return {
                "extracted_transactions": transactions,
                "messages": [f"Extracted {len(transactions)} transactions"]
            }
        
        except Exception as e:
            return {"error": str(e), "messages": [f"Error extracting transactions: {str(e)}"]}
    
    def categorize_transactions(self, state: FinancialState) -> dict:
        """Categorize transactions using LLM"""
        try:
            transactions = state.get("extracted_transactions", [])
            
            if not transactions:
                return {"error": "No transactions to categorize", "messages": ["Error: No transactions"]}
            
            # Prepare transaction descriptions for categorization
            descriptions = [t['description'] for t in transactions]
            
            # Use LLM to categorize in batches
            categorized = []
            batch_size = 50
            
            for i in range(0, len(transactions), batch_size):
                batch = transactions[i:i+batch_size]
                batch_descriptions = [t['description'] for t in batch]
                
                prompt = f"""Categorize the following financial transactions into these categories:
                
Categories:
- Revenue/Sales
- Cost of Goods Sold
- Operating Expenses (Salaries, Rent, Utilities, Marketing, etc.)
- Assets (Equipment, Inventory, Receivables)
- Liabilities (Loans, Payables)
- Equity
- Other Income
- Other Expenses

Transactions:
{json.dumps(batch_descriptions, indent=2)}

Return a JSON array where each element is the category name for the corresponding transaction.
Only return the JSON array, no additional text.
"""
                
                response = self.llm.invoke(prompt)
                categories_text = response.content.strip()
                
                # Parse response
                if categories_text.startswith('```'):
                    categories_text = categories_text.split('```')[1]
                    if categories_text.startswith('json'):
                        categories_text = categories_text[4:]
                
                try:
                    categories = json.loads(categories_text)
                except:
                    # Fallback categorization
                    categories = ["Other Expenses" if t['debit'] > 0 else "Revenue/Sales" for t in batch]
                
                for j, transaction in enumerate(batch):
                    transaction['category'] = categories[j] if j < len(categories) else "Uncategorized"
                    categorized.append(transaction)
            
            return {
                "categorized_transactions": categorized,
                "messages": [f"Categorized {len(categorized)} transactions"]
            }
        
        except Exception as e:
            return {"error": str(e), "messages": [f"Error categorizing: {str(e)}"]}
    
    def create_vector_store(self, state: FinancialState) -> dict:
        """Create vector store for RAG retrieval"""
        try:
            transactions = state.get("categorized_transactions", [])
            
            # Create documents from transactions
            documents = []
            for i, txn in enumerate(transactions):
                content = f"""
                Date: {txn['date']}
                Description: {txn['description']}
                Category: {txn['category']}
                Debit: ₹{txn['debit']:,.2f}
                Credit: ₹{txn['credit']:,.2f}
                Balance: ₹{txn['balance']:,.2f}
                """
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'transaction_id': i,
                        'category': txn['category'],
                        'date': str(txn['date']),
                        'amount': txn['debit'] if txn['debit'] > 0 else txn['credit']
                    }
                )
                documents.append(doc)
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            return {"messages": ["Vector store created successfully"]}
        
        except Exception as e:
            return {"error": str(e), "messages": [f"Error creating vector store: {str(e)}"]}
    
    def calculate_profit_loss(self, state: FinancialState) -> dict:
        """Calculate Profit & Loss statement"""
        try:
            transactions = state.get("categorized_transactions", [])
            
            # Calculate revenue
            revenue = sum(t['credit'] for t in transactions if 'revenue' in t['category'].lower() or 'sales' in t['category'].lower())
            
            # Calculate COGS
            cogs = sum(t['debit'] for t in transactions if 'cost of goods' in t['category'].lower())
            
            # Calculate operating expenses
            operating_expenses = sum(t['debit'] for t in transactions if 'operating' in t['category'].lower())
            
            # Other income and expenses
            other_income = sum(t['credit'] for t in transactions if 'other income' in t['category'].lower())
            other_expenses = sum(t['debit'] for t in transactions if 'other expense' in t['category'].lower())
            
            # Calculate totals
            gross_profit = revenue - cogs
            operating_income = gross_profit - operating_expenses
            net_income = operating_income + other_income - other_expenses
            
            profit_loss = {
                "Revenue": revenue,
                "Cost of Goods Sold": cogs,
                "Gross Profit": gross_profit,
                "Operating Expenses": operating_expenses,
                "Operating Income": operating_income,
                "Other Income": other_income,
                "Other Expenses": other_expenses,
                "Net Income": net_income,
                "Report Date": datetime.now().strftime("%Y-%m-%d")
            }
            
            return {
                "profit_loss": profit_loss,
                "messages": [f"Profit & Loss calculated. Net Income: ₹{net_income:,.2f}"]
            }
        
        except Exception as e:
            return {"error": str(e), "messages": [f"Error calculating P&L: {str(e)}"]}
    
    def calculate_balance_sheet(self, state: FinancialState) -> dict:
        """Calculate Balance Sheet"""
        try:
            transactions = state.get("categorized_transactions", [])
            
            # Assets
            current_assets = sum(t['debit'] for t in transactions if 'asset' in t['category'].lower())
            cash = transactions[-1]['balance'] if transactions else 0
            total_assets = current_assets + cash
            
            # Liabilities
            current_liabilities = sum(t['credit'] for t in transactions if 'liabilit' in t['category'].lower())
            
            # Equity
            equity_contributions = sum(t['credit'] for t in transactions if 'equity' in t['category'].lower())
            retained_earnings = state.get("profit_loss", {}).get("Net Income", 0)
            total_equity = equity_contributions + retained_earnings
            
            total_liabilities_equity = current_liabilities + total_equity
            
            balance_sheet = {
                "Assets": {
                    "Cash": cash,
                    "Other Current Assets": current_assets,
                    "Total Assets": total_assets
                },
                "Liabilities": {
                    "Current Liabilities": current_liabilities,
                    "Total Liabilities": current_liabilities
                },
                "Equity": {
                    "Equity Contributions": equity_contributions,
                    "Retained Earnings": retained_earnings,
                    "Total Equity": total_equity
                },
                "Total Liabilities & Equity": total_liabilities_equity,
                "Report Date": datetime.now().strftime("%Y-%m-%d")
            }
            
            return {
                "balance_sheet": balance_sheet,
                "messages": [f"Balance Sheet calculated. Total Assets: ₹{total_assets:,.2f}"]
            }
        
        except Exception as e:
            return {"error": str(e), "messages": [f"Error calculating Balance Sheet: {str(e)}"]}
    
    def calculate_cash_flow(self, state: FinancialState) -> dict:
        """Calculate Cash Flow statement"""
        try:
            transactions = state.get("categorized_transactions", [])
            
            if not transactions:
                return {"cash_flow": {}, "messages": ["No transactions for cash flow"]}
            
            # Operating activities
            cash_from_operations = sum(t['credit'] - t['debit'] for t in transactions 
                                      if 'revenue' in t['category'].lower() or 'expense' in t['category'].lower())
            
            # Investing activities
            cash_from_investing = sum(t['credit'] - t['debit'] for t in transactions 
                                     if 'asset' in t['category'].lower())
            
            # Financing activities
            cash_from_financing = sum(t['credit'] - t['debit'] for t in transactions 
                                     if 'liabilit' in t['category'].lower() or 'equity' in t['category'].lower())
            
            net_cash_flow = cash_from_operations + cash_from_investing + cash_from_financing
            
            cash_flow = {
                "Operating Activities": cash_from_operations,
                "Investing Activities": cash_from_investing,
                "Financing Activities": cash_from_financing,
                "Net Cash Flow": net_cash_flow,
                "Ending Cash Balance": transactions[-1]['balance'] if transactions else 0,
                "Report Date": datetime.now().strftime("%Y-%m-%d")
            }
            
            return {
                "cash_flow": cash_flow,
                "messages": [f"Cash Flow calculated. Net Cash Flow: ₹{net_cash_flow:,.2f}"]
            }
        
        except Exception as e:
            return {"error": str(e), "messages": [f"Error calculating Cash Flow: {str(e)}"]}
    
    def generate_analysis(self, state: FinancialState) -> dict:
        """Generate comprehensive financial analysis using RAG"""
        try:
            profit_loss = state.get("profit_loss", {})
            balance_sheet = state.get("balance_sheet", {})
            cash_flow = state.get("cash_flow", {})
            
            # Use RAG to get relevant context
            query = "Provide insights on revenue trends, expense patterns, and financial health"
            relevant_docs = self.vector_store.similarity_search(query, k=10)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate analysis using LLM
            prompt = f"""Based on the following financial data, provide a comprehensive analysis:

PROFIT & LOSS STATEMENT:
{json.dumps(profit_loss, indent=2)}

BALANCE SHEET:
{json.dumps(balance_sheet, indent=2)}

CASH FLOW STATEMENT:
{json.dumps(cash_flow, indent=2)}

TRANSACTION CONTEXT:
{context}

Provide a detailed financial analysis covering:
1. Overall financial health
2. Profitability analysis
3. Liquidity position
4. Key risks and opportunities
5. Recommendations for improvement

Keep the analysis professional and actionable.
"""
            
            response = self.llm.invoke(prompt)
            analysis = response.content
            
            return {
                "analysis_summary": analysis,
                "messages": ["Financial analysis completed successfully"]
            }
        
        except Exception as e:
            return {"error": str(e), "messages": [f"Error generating analysis: {str(e)}"]}
    
    def _parse_date(self, date_str):
        """Parse date from various formats"""
        if pd.isna(date_str):
            return None
        
        try:
            return pd.to_datetime(date_str).strftime("%Y-%m-%d")
        except:
            return str(date_str)
    
    def process_bank_statement(self, file_path: str) -> dict:
        """Main method to process bank statement"""
        initial_state = {
            "uploaded_file": file_path,
            "extracted_transactions": [],
            "categorized_transactions": [],
            "profit_loss": {},
            "balance_sheet": {},
            "cash_flow": {},
            "analysis_summary": "",
            "error": "",
            "messages": []
        }
        
        # Run the workflow
        result = self.app.invoke(initial_state)
        
        return {
            "status": "success" if not result.get("error") else "error",
            "profit_loss": result.get("profit_loss", {}),
            "balance_sheet": result.get("balance_sheet", {}),
            "cash_flow": result.get("cash_flow", {}),
            "analysis": result.get("analysis_summary", ""),
            "messages": result.get("messages", []),
            "error": result.get("error", "")
        }

# Initialize system
rag_system = FinancialRAGSystem()
