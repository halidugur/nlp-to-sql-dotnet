using Microsoft.EntityFrameworkCore;
using System;
using stajprojem.Data;


var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection")));
builder.Services.AddHttpClient<stajprojem.Services.PythonNlpSqlApiService>();
// Program.cs veya Startup.cs'de (Program.cs için .NET 6 ve üstü)
builder.Services.AddHttpClient("NlpSqlApi", client =>
{
    client.BaseAddress = new Uri("http://127.0.0.1:8000/"); // FastAPI endpoint adresi!
});
builder.Services.AddSingleton<stajprojem.Services.AdoSqlExecutorService>();






var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
