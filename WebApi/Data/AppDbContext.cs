using Microsoft.EntityFrameworkCore;
using stajprojem.Models;

namespace stajprojem.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        public DbSet<Customer> Customers { get; set; }
        public DbSet<Category> Categories { get; set; }
        public DbSet<Employee> Employees { get; set; }
        public DbSet<Order> Orders { get; set; }
        public DbSet<OrderDetail> OrderDetails { get; set; }
        public DbSet<Product> Products { get; set; }
        public DbSet<PurchaseOrder> PurchaseOrders { get; set; }
        public DbSet<Supplier> Suppliers { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<Customer>().ToTable("customers");
            modelBuilder.Entity<Category>().ToTable("categories");
            modelBuilder.Entity<Employee>().ToTable("employees");
            modelBuilder.Entity<Order>().ToTable("orders");
            modelBuilder.Entity<OrderDetail>().ToTable("order_details");
            modelBuilder.Entity<Product>().ToTable("products");
            modelBuilder.Entity<PurchaseOrder>().ToTable("purchase_orders");
            modelBuilder.Entity<Supplier>().ToTable("suppliers");
        }
    }
}
